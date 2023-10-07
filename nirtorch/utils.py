def sanitize_name(name: str):
    """Sanitize the name of a node."""
    return name.replace(".", "_")
