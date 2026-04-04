from __future__ import annotations

from .domain import Ability

_CONCISE_SUFFIX = "Answer shortly unless requested to elaborate."

PERSONA_PRESETS: dict[str, str] = {
    "Creator Studio": f"You are a creative operations director helping a content team turn ideas into reusable production assets. {_CONCISE_SUFFIX}",
    "Incident Desk": f"You are an analyst in an operations room who needs crisp summaries, evidence framing, and action-oriented outputs. {_CONCISE_SUFFIX}",
    "Learning Lab": f"You are a tutor and explainer who helps users understand what the model sees, hears, or infers. {_CONCISE_SUFFIX}",
    "Freeplay": f"You are a flexible sandbox assistant. Adapt to the user's goal while staying explicit about uncertainty. {_CONCISE_SUFFIX}",
    "Custom": "",
}


ABILITY_SPECS = {
    Ability.TEXT_TO_TEXT: ("Native", "Direct Gemma text generation and reasoning."),
    Ability.IMAGE_TO_TEXT: ("Native", "Direct Gemma image understanding with text output."),
    Ability.AUDIO_TO_TEXT: ("Native", "Direct Gemma audio understanding on E2B and E4B checkpoints."),
    Ability.VIDEO_TO_TEXT: ("Experimental", "Frame-sampled video analysis routed through Gemma image understanding."),
    Ability.TEXT_TO_IMAGE: ("Simulated", "Gemma creates a prompt pack and creative brief for an external image model."),
    Ability.TEXT_TO_VIDEO: ("Simulated", "Gemma creates a storyboard and shot plan for an external video model."),
    Ability.TEXT_TO_AUDIO: ("Simulated", "Gemma creates narration, voice, and sound design instructions for an external audio model."),
}


def build_text_prompt(user_prompt: str) -> str:
    return user_prompt


def build_image_prompt(user_prompt: str) -> str:
    return user_prompt or "Describe the image clearly, then list the most important details and likely context."


def build_audio_prompt(user_prompt: str) -> str:
    return user_prompt or (
        "Transcribe the following speech segment in its original language. "
        "Only output the transcription, with no newlines. "
        "When transcribing numbers, write digits instead of spelled-out words."
    )


def build_video_prompt(user_prompt: str) -> str:
    return user_prompt or (
        "Analyze the sampled frames from this video. Describe the setting, actions, sequence of events, "
        "and anything uncertain or missing because only representative frames are available."
    )


def build_simulation_prompt(ability: Ability, user_prompt: str) -> str:
    base_prompt = user_prompt or "Create a strong production-ready plan from this concept."
    instructions = {
        Ability.TEXT_TO_IMAGE: (
            "Return sections titled Concept, Final Prompt, Negative Prompt, Style Notes, Composition Notes, "
            "and Model Handoff Notes."
        ),
        Ability.TEXT_TO_VIDEO: (
            "Return sections titled Concept, One-Sentence Hook, Storyboard, Shot List, Motion Notes, Audio Notes, "
            "and Model Handoff Notes."
        ),
        Ability.TEXT_TO_AUDIO: (
            "Return sections titled Concept, Final Script, Voice Direction, Sound Design, Timing Plan, "
            "and Model Handoff Notes."
        ),
    }[ability]
    return (
        f"This mode is a simulation. Gemma is not rendering media. "
        f"Produce a structured planning artifact for {ability.value}. {instructions}\n\n"
        f"User request:\n{base_prompt}"
    )