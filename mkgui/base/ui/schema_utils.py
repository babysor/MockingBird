from typing import Dict


def resolve_reference(reference: str, references: Dict) -> Dict:
    return references[reference.split("/")[-1]]


def get_single_reference_item(property: Dict, references: Dict) -> Dict:
    # Ref can either be directly in the properties or the first element of allOf
    reference = property.get("$ref")
    if reference is None:
        reference = property["allOf"][0]["$ref"]
    return resolve_reference(reference, references)


def is_single_string_property(property: Dict) -> bool:
    return property.get("type") == "string"


def is_single_datetime_property(property: Dict) -> bool:
    if property.get("type") != "string":
        return False
    return property.get("format") in ["date-time", "time", "date"]


def is_single_boolean_property(property: Dict) -> bool:
    return property.get("type") == "boolean"


def is_single_number_property(property: Dict) -> bool:
    return property.get("type") in ["integer", "number"]


def is_single_file_property(property: Dict) -> bool:
    if property.get("type") != "string":
        return False
    # TODO: binary?
    return property.get("format") == "byte"


def is_multi_enum_property(property: Dict, references: Dict) -> bool:
    if property.get("type") != "array":
        return False

    if property.get("uniqueItems") is not True:
        # Only relevant if it is a set or other datastructures with unique items
        return False

    try:
        _ = resolve_reference(property["items"]["$ref"], references)["enum"]
        return True
    except Exception:
        return False


def is_single_enum_property(property: Dict, references: Dict) -> bool:
    try:
        _ = get_single_reference_item(property, references)["enum"]
        return True
    except Exception:
        return False


def is_single_dict_property(property: Dict) -> bool:
    if property.get("type") != "object":
        return False
    return "additionalProperties" in property


def is_single_reference(property: Dict) -> bool:
    if property.get("type") is not None:
        return False

    return bool(property.get("$ref"))


def is_multi_file_property(property: Dict) -> bool:
    if property.get("type") != "array":
        return False

    if property.get("items") is None:
        return False

    try:
        # TODO: binary
        return property["items"]["format"] == "byte"
    except Exception:
        return False


def is_single_object(property: Dict, references: Dict) -> bool:
    try:
        object_reference = get_single_reference_item(property, references)
        if object_reference["type"] != "object":
            return False
        return "properties" in object_reference
    except Exception:
        return False


def is_property_list(property: Dict) -> bool:
    if property.get("type") != "array":
        return False

    if property.get("items") is None:
        return False

    try:
        return property["items"]["type"] in ["string", "number", "integer"]
    except Exception:
        return False


def is_object_list_property(property: Dict, references: Dict) -> bool:
    if property.get("type") != "array":
        return False

    try:
        object_reference = resolve_reference(property["items"]["$ref"], references)
        if object_reference["type"] != "object":
            return False
        return "properties" in object_reference
    except Exception:
        return False
