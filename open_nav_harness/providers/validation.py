"""Tool-argument validation shared by every model/tool loop."""

import jsonschema

# Tools validate with pydantic strict mode, where 2.0 is not an integer. Batch checks must
# agree, or a call could pass here and fail only after earlier calls in its batch moved.
StrictValidator = jsonschema.validators.extend(
    jsonschema.Draft202012Validator,
    type_checker=jsonschema.Draft202012Validator.TYPE_CHECKER.redefine(
        "integer", lambda _, value: isinstance(value, int) and not isinstance(value, bool)
    ),
)
