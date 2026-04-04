from gemma_sandbox.domain import Ability
from gemma_sandbox.prompts import ABILITY_SPECS, PERSONA_PRESETS, build_simulation_prompt, build_text_prompt


def test_text_prompt_returns_user_input() -> None:
    prompt = build_text_prompt("Summarize this scene.")

    assert prompt == "Summarize this scene."


def test_persona_presets_include_concise_instruction() -> None:
    for name, text in PERSONA_PRESETS.items():
        if name == "Custom":
            assert text == ""
        else:
            assert "Answer shortly unless requested to elaborate" in text


def test_simulation_prompt_marks_non_native_mode() -> None:
    prompt = build_simulation_prompt(Ability.TEXT_TO_VIDEO, "Create a teaser for a robot race.")

    assert "This mode is a simulation" in prompt
    assert "text-to-video" in prompt
    assert "Storyboard" in prompt


def test_ability_specs_label_simulated_modes() -> None:
    support_level, summary = ABILITY_SPECS[Ability.TEXT_TO_AUDIO]

    assert support_level == "Simulated"
    assert "external audio model" in summary