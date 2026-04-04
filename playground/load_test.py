#!/usr/bin/env python3
"""
Concurrent Load Testing for Model Serving API

Tests the /generate endpoint with multiple concurrent users to identify bottlenecks,
measure throughput under load, and validate production readiness.

Usage:
    python load_test.py load_scenarios.json --concurrent-users 50 --duration 60
    python load_test.py real-benchmark-scenarios.json --concurrent-users 10 --ramp-up 5
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass, field
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List
import sys

try:
    import aiohttp
except ImportError:
    print("❌ aiohttp is required for load testing. Install with: pip install aiohttp")
    sys.exit(1)

# Add model-serving to path for shared types
if __package__ in {None, ""}:
    PACKAGE_ROOT = Path(__file__).resolve().parents[1]
    MODEL_SERVING_SRC = PACKAGE_ROOT / "model-serving" / "src"
    if str(MODEL_SERVING_SRC) not in sys.path:
        sys.path.insert(0, str(MODEL_SERVING_SRC))

try:
    from gemma_serving.domain import RequestMode
except ImportError:
    # Fallback for standalone usage
    from enum import Enum
    class RequestMode(Enum):
        TEXT_ONLY = "text-only"
        MULTIMODAL = "multimodal"


@dataclass(frozen=True)
class LoadTestScenario:
    """Scenario definition for concurrent load testing"""
    name: str
    model_label: str
    request_mode: RequestMode
    concurrent_users: int = 10
    duration_seconds: int = 60
    ramp_up_seconds: int = 0
    messages: List[Dict[str, str]] = field(default_factory=list)
    max_length: int = 256
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.concurrent_users <= 0:
            raise ValueError("concurrent_users must be positive")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.ramp_up_seconds < 0:
            raise ValueError("ramp_up_seconds must be non-negative")


@dataclass(frozen=True)
class RequestResult:
    """Individual request result"""
    started_at: float
    completed_at: float
    status_code: int
    success: bool
    response_size: int = 0
    error_message: str | None = None

    @property
    def latency_seconds(self) -> float:
        return self.completed_at - self.started_at


@dataclass(frozen=True)
class LoadTestResult:
    """Complete load test results"""
    scenario_name: str
    model_label: str
    concurrent_users: int
    duration_seconds: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    requests_per_second: float
    average_latency_seconds: float
    p50_latency_seconds: float
    p95_latency_seconds: float
    p99_latency_seconds: float
    min_latency_seconds: float
    max_latency_seconds: float
    error_rate_percent: float
    total_response_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LoadTester:
    """Concurrent HTTP load testing orchestrator"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        self.logger = logging.getLogger(__name__)

    async def run_load_test(self, scenario: LoadTestScenario) -> LoadTestResult:
        """Execute concurrent load test for scenario"""
        self.logger.info(
            f"🚀 Starting load test '{scenario.name}': {scenario.concurrent_users} users, {scenario.duration_seconds}s"
        )

        # Health check first
        await self._health_check()

        # Prepare request payload
        request_payload = {
            "messages": scenario.messages or [
                {"role": "user", "content": "Generate a short response for load testing."}
            ],
            "max_length": scenario.max_length
        }

        # Run concurrent requests
        results = await self._execute_concurrent_requests(
            scenario, request_payload
        )

        # Calculate metrics
        return self._calculate_metrics(scenario, results)

    async def _health_check(self) -> None:
        """Verify server is responding"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health", timeout=5) as response:
                    if response.status != 200:
                        raise Exception(f"Health check failed: {response.status}")
                    self.logger.info("✅ Server health check passed")
        except Exception as e:
            self.logger.error(f"❌ Server health check failed: {e}")
            raise

    async def _execute_concurrent_requests(
        self, 
        scenario: LoadTestScenario, 
        request_payload: Dict[str, Any]
    ) -> List[RequestResult]:
        """Execute concurrent requests with optional ramp-up"""
        results: List[RequestResult] = []
        start_time = time.time()
        
        # Create semaphore to limit concurrent connections
        semaphore = asyncio.Semaphore(scenario.concurrent_users)
        
        # Calculate request timing
        total_duration = scenario.duration_seconds + scenario.ramp_up_seconds
        
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=scenario.concurrent_users * 2),
            timeout=aiohttp.ClientTimeout(total=300)  # 5 min timeout per request
        ) as session:
            tasks = []
            
            # Create worker tasks
            for user_id in range(scenario.concurrent_users):
                # Calculate ramp-up delay
                if scenario.ramp_up_seconds > 0:
                    delay = (user_id / scenario.concurrent_users) * scenario.ramp_up_seconds
                else:
                    delay = 0
                
                task = asyncio.create_task(
                    self._user_worker(
                        session, semaphore, request_payload, 
                        user_id, delay, start_time, total_duration
                    )
                )
                tasks.append(task)
            
            # Wait for all workers to complete
            worker_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Flatten results
            for worker_result in worker_results:
                if isinstance(worker_result, list):
                    results.extend(worker_result)
                elif isinstance(worker_result, Exception):
                    self.logger.error(f"Worker failed: {worker_result}")

        # Include all completed requests. Filtering by start or completion time
        # is counterproductive for high-latency backends like Gemma where a single
        # inference can span the entire ramp-up window. Ramp-up is purely a
        # request-dispatch delay — all results are valid measurements.
        test_start = start_time + scenario.ramp_up_seconds
        filtered_results = results
        ramp_up_only_count = sum(1 for r in results if r.completed_at < test_start)
        if ramp_up_only_count:
            self.logger.info(
                f"ℹ️  {ramp_up_only_count} of {len(results)} requests also "
                f"completed within the ramp-up window"
            )

        self.logger.info(
            f"📊 Completed: {len(filtered_results)} requests "
            f"({len(results)} total including ramp-up)"
        )
        
        return filtered_results

    async def _user_worker(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        request_payload: Dict[str, Any],
        user_id: int,
        start_delay: float,
        test_start_time: float,
        total_duration: float
    ) -> List[RequestResult]:
        """Simulate individual user sending requests"""
        results = []
        
        # Wait for ramp-up delay
        if start_delay > 0:
            await asyncio.sleep(start_delay)
        
        end_time = test_start_time + total_duration
        
        while time.time() < end_time:
            async with semaphore:
                started_at = time.time()
                
                # Stop if we've passed the end time
                if started_at >= end_time:
                    break
                
                try:
                    async with session.post(
                        f"{self.base_url}/generate",
                        json=request_payload,
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        completed_at = time.time()
                        response_text = await response.text()
                        
                        results.append(RequestResult(
                            started_at=started_at,
                            completed_at=completed_at,
                            status_code=response.status,
                            success=200 <= response.status < 300,
                            response_size=len(response_text.encode('utf-8')),
                        ))
                        
                except Exception as e:
                    completed_at = time.time()
                    results.append(RequestResult(
                        started_at=started_at,
                        completed_at=completed_at,
                        status_code=0,
                        success=False,
                        error_message=str(e)
                    ))
        
        return results

    def _calculate_metrics(
        self, 
        scenario: LoadTestScenario, 
        results: List[RequestResult]
    ) -> LoadTestResult:
        """Calculate comprehensive test metrics"""
        if not results:
            raise ValueError("No results to analyze")
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        latencies = [r.latency_seconds for r in successful_results]
        
        if latencies:
            avg_latency = statistics.fmean(latencies)
            p50_latency = self._percentile(latencies, 0.50)
            p95_latency = self._percentile(latencies, 0.95)
            p99_latency = self._percentile(latencies, 0.99)
            min_latency = min(latencies)
            max_latency = max(latencies)
        else:
            avg_latency = p50_latency = p95_latency = p99_latency = 0.0
            min_latency = max_latency = 0.0
        
        total_requests = len(results)
        successful_requests = len(successful_results)
        failed_requests = len(failed_results)
        
        # Calculate RPS (exclude ramp-up period)
        requests_per_second = total_requests / scenario.duration_seconds
        
        error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0.0
        
        total_response_bytes = sum(r.response_size for r in results)
        
        return LoadTestResult(
            scenario_name=scenario.name,
            model_label=scenario.model_label,
            concurrent_users=scenario.concurrent_users,
            duration_seconds=scenario.duration_seconds,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            requests_per_second=requests_per_second,
            average_latency_seconds=avg_latency,
            p50_latency_seconds=p50_latency,
            p95_latency_seconds=p95_latency,
            p99_latency_seconds=p99_latency,
            min_latency_seconds=min_latency,
            max_latency_seconds=max_latency,
            error_rate_percent=error_rate,
            total_response_bytes=total_response_bytes,
        )

    def _percentile(self, values: List[float], fraction: float) -> float:
        """Calculate percentile from sorted values"""
        if not values:
            return 0.0
        ordered = sorted(values)
        index = max(0, round((len(ordered) - 1) * fraction))
        return ordered[index]


def load_scenarios(file_path: Path) -> List[LoadTestScenario]:
    """Load scenarios from JSON file"""
    raw_payload = json.loads(file_path.read_text(encoding="utf-8"))
    scenarios = []
    
    for item in raw_payload:
        # Convert existing benchmark scenarios or load new format
        if "concurrent_users" in item:
            # New load test format
            scenario = LoadTestScenario(
                name=item["name"],
                model_label=item["model_label"],
                request_mode=RequestMode(item["request_mode"]),
                concurrent_users=item["concurrent_users"],
                duration_seconds=item.get("duration_seconds", 60),
                ramp_up_seconds=item.get("ramp_up_seconds", 0),
                messages=item.get("messages", []),
                max_length=item.get("max_length", 256),
                metadata=item.get("metadata", {}),
            )
        else:
            # Convert from existing benchmark format
            scenario = LoadTestScenario(
                name=item["name"],
                model_label=item["model_label"],
                request_mode=RequestMode(item["request_mode"]),
                concurrent_users=10,  # Default for converted scenarios
                duration_seconds=30,  # Default for converted scenarios
                messages=[{"role": "user", "content": "Test message for load testing"}],
                max_length=item["metadata"].get("max_new_tokens", 256),
                metadata=item.get("metadata", {}),
            )
        
        scenarios.append(scenario)
    
    return scenarios


def print_results(results: List[LoadTestResult]) -> None:
    """Print formatted test results"""
    print("\n" + "=" * 80)
    print("🏁 LOAD TEST RESULTS")
    print("=" * 80)
    
    for result in results:
        print(f"\n📊 {result.scenario_name} ({result.model_label})")
        print("-" * 60)
        print(f"   👥 Concurrent Users: {result.concurrent_users}")
        print(f"   ⏱️  Duration: {result.duration_seconds}s")
        print(f"   📈 Total Requests: {result.total_requests}")
        print(f"   ✅ Success Rate: {(result.successful_requests/result.total_requests*100):.1f}%")
        print(f"   🚀 Requests/sec: {result.requests_per_second:.2f}")
        print(f"   ⚡ Avg Latency: {result.average_latency_seconds:.3f}s")
        print(f"   📊 P50/P95/P99: {result.p50_latency_seconds:.3f}s / "
              f"{result.p95_latency_seconds:.3f}s / {result.p99_latency_seconds:.3f}s")
        print(f"   📦 Total Data: {result.total_response_bytes / 1024:.1f} KB")
        
        if result.error_rate_percent > 0:
            print(f"   ❌ Error Rate: {result.error_rate_percent:.1f}%")


async def main() -> int:
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Run concurrent load tests against model-serving API")
    parser.add_argument("scenario_file", type=Path, help="JSON file with load test scenarios")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Model serving base URL")
    parser.add_argument("--concurrent-users", type=int, help="Override concurrent users for all scenarios")
    parser.add_argument("--duration", type=int, help="Override duration for all scenarios")
    parser.add_argument("--ramp-up", type=int, help="Override ramp-up time for all scenarios")
    parser.add_argument("--output", type=Path, help="Save results to JSON file")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True
    )
    
    # Load scenarios
    scenarios = load_scenarios(args.scenario_file)
    
    # Apply CLI overrides
    if args.concurrent_users is not None:
        scenarios = [
            LoadTestScenario(
                name=s.name,
                model_label=s.model_label,
                request_mode=s.request_mode,
                concurrent_users=args.concurrent_users,
                duration_seconds=args.duration or s.duration_seconds,
                ramp_up_seconds=args.ramp_up or s.ramp_up_seconds,
                messages=s.messages,
                max_length=s.max_length,
                metadata=s.metadata,
            )
            for s in scenarios
        ]
    
    # Run load tests
    tester = LoadTester(args.base_url)
    results = []
    
    for scenario in scenarios:
        try:
            result = await tester.run_load_test(scenario)
            results.append(result)
        except Exception as e:
            logging.error(f"Load test failed for {scenario.name}: {e}")
            continue
    
    # Output results
    if results:
        print_results(results)
        
        if args.output:
            output_data = [r.to_dict() for r in results]
            args.output.write_text(json.dumps(output_data, indent=2))
            print(f"\n💾 Results saved to {args.output}")
    else:
        print("❌ No successful load tests completed")
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Load test interrupted by user")
        sys.exit(1)