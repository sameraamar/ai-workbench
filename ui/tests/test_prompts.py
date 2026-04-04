from gemma_sandbox.domain import Ability
from gemma_sandbox.prompts import ABILITY_SPECS, build_simulation_prompt, build_text_prompt


def test_text_prompt_includes_selected_preset() -> None:
    prompt = build_text_prompt("Summarize this scene.", "Incident Desk")

    assert "operations room" in prompt
    assert "Summarize this scene." in prompt


def test_simulation_prompt_marks_non_native_mode() -> None:
    prompt = build_simulation_prompt(Ability.TEXT_TO_VIDEO, "Create a teaser for a robot race.", "Creator Studio")

    assert "This mode is a simulation" in prompt
    assert "text-to-video" in prompt
    assert "Storyboard" in prompt


def test_ability_specs_label_simulated_modes() -> None:
    support_level, summary = ABILITY_SPECS[Ability.TEXT_TO_AUDIO]

    assert support_level == "Simulated"
    assert "external audio model" in summary