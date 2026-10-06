from ron.tools import ToolRegistry

def test_tool_registry_requires_explicit_registration():
    tools = ToolRegistry()
    tools.register("echo", lambda args: {"value": args["value"]})
    assert tools.execute("echo", {"value": "ok"})["value"] == "ok"
    assert tools.describe() == [{"name": "echo"}]