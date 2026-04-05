from model_serving.config import ServingConfig
from model_serving.gateway import ModelGateway
from model_serving.app import AttributeExtractionRequest, ListingRewriteRequest


class FakeModelService:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.messages = None

    def generate(self, messages, settings):
        self.messages = messages
        return {
            "text": self._response_text,
            "input_token_count": 10,
            "output_token_count": 20,
            "total_token_count": 30,
        }


def test_rewrite_listing_uses_model_json_response() -> None:
    gateway = ModelGateway(
        config=ServingConfig(),
        model_service=FakeModelService('{"title": "eBay title", "description": "Cleaned description"}'),
    )

    result = gateway.rewrite_listing(
        ListingRewriteRequest(
            title="old title",
            description="old description",
            marketplace="ebay",
            category_hint="jewelry",
        )
    )

    assert result["title"] == "eBay title"
    assert result["description"] == "Cleaned description"
    assert result["model_id"] == "google/gemma-4-E2B-it"


def test_extract_attributes_limits_images_and_returns_metadata() -> None:
    gateway = ModelGateway(
        config=ServingConfig(),
        model_service=FakeModelService('{"suggested_attributes": {"material": "silver"}}'),
    )

    result = gateway.extract_attributes(
        AttributeExtractionRequest(
            image_urls=["img-1.jpg", "img-2.jpg", "img-3.jpg"],
            attribute_hints=["material", "stone"],
            max_images=2,
        )
    )

    assert result["inspected_images"] == ["img-1.jpg", "img-2.jpg"]
    assert result["suggested_attributes"]["material"] == "silver"
