from gemma_sandbox.prompts import PERSONA_PRESETS


def test_persona_presets_include_concise_instruction() -> None:
    for name, text in PERSONA_PRESETS.items():
        if name == "Custom":
            assert text == ""
        else:
            assert len(text) > 0


def test_planning_personas_exist() -> None:
    assert "Image Prompt Pack" in PERSONA_PRESETS
    assert "Video Storyboard" in PERSONA_PRESETS
    assert "Audio Script" in PERSONA_PRESETS


def test_image_persona_mentions_sections() -> None:
    text = PERSONA_PRESETS["Image Prompt Pack"]
    assert "Final Prompt" in text
    assert "Concept" in text


def test_storyboard_persona_mentions_sections() -> None:
    text = PERSONA_PRESETS["Video Storyboard"]
    assert "Storyboard" in text
    assert "Shot List" in text