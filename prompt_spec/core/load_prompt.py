
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
from .prompt_template import PromptTemplate, PromptExample, PromptType
from .utils import load_pydantic_class
import yaml
import json
from typing import Type

class RawPromptExample(BaseModel):
    input: str
    output: str  # JSON string

class RawPromptSpec(BaseModel):
    name: str
    prompt_type: PromptType
    system: Optional[str] = None
    instruction: str
    output_model: str  # symbolic name
    examples: List[RawPromptExample] = []

def load_prompt_template(
    prompt_path: Path,
    *,
    linkml_schema: Path | None = None,
    module_path: Path | None = None,
) -> PromptTemplate:
    """
    Load a prompt YAML, resolve its output_model to a Pydantic class,
    and validate examples against that model.
    """

    raw_data = yaml.safe_load(prompt_path.read_text())
    raw = RawPromptSpec.model_validate(raw_data)

    # Resolve output_model → Pydantic class
    output_model: Type[BaseModel] = load_pydantic_class(
        raw.output_model,
        linkml_schema=linkml_schema,
        module_path=module_path,
    )

    # Parse and validate examples
    examples: list[PromptExample] = []
    for ex in raw.examples:
        data = json.loads(ex.output)
        validated = output_model.model_validate(data)
        examples.append(
            PromptExample(
                input=ex.input,
                output=validated,
            )
        )

    return PromptTemplate(
        name=raw.name,
        prompt_type=raw.prompt_type,
        system=raw.system,
        instruction=raw.instruction,
        output_model=output_model,
        examples=examples,
    )
