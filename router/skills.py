from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class Skill:
    name: str
    description: str
    parameters: dict[str, Any]

    async def execute(self, **kwargs: Any) -> str:
        raise NotImplementedError


class WebSearchSkill(Skill):
    async def execute(self, query: str, **kwargs: Any) -> str:
        """Perform a web search using DuckDuckGo API."""
        if not query:
            return "Error: query parameter is required for web search."
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_html": 1},
                )
                response.raise_for_status()
                data = response.json()

                # Extract relevant snippets
                results = []
                if data.get("AbstractText"):
                    results.append(f"Abstract: {data['AbstractText']}")

                for result in data.get("RelatedTopics", []):
                    if "Text" in result:
                        results.append(result["Text"])

                if not results:
                    return "No results found."

                return "\n".join(results)
        except Exception as e:
            return f"Error performing web search: {e}"


class CalculatorSkill(Skill):
    async def execute(self, expression: str, **kwargs: Any) -> str:
        """Safely evaluate a mathematical expression."""
        import operator

        allowed_operators = {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv,
            "^": operator.pow,
        }

        try:
            # Basic shunting-yard/RPN would be safer, but for now we'll do a simple split
            # This is NOT production safe for complex expressions
            parts = expression.replace(" ", "")
            for op_symbol, op_func in allowed_operators.items():
                if op_symbol in parts:
                    left, right = parts.split(op_symbol, 1)
                    return str(op_func(float(left), float(right)))
            return "Error: Invalid or unsupported expression."
        except Exception as e:
            return f"Error evaluating expression: {e}"


class SkillsRegistry:
    def __init__(self):
        self._skills: dict[str, Skill] = {}
        self._initialize_skills()

    def _initialize_skills(self) -> None:
        self.register(
            WebSearchSkill(
                name="web_search",
                description="Get information from the web. Use for current events or specific knowledge.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                    },
                    "required": ["query"],
                },
            )
        )
        self.register(
            CalculatorSkill(
                name="calculator",
                description="Evaluate mathematical expressions.",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The expression to evaluate",
                        },
                    },
                    "required": ["expression"],
                },
            )
        )

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def list_skills(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": skill.name,
                    "description": skill.description,
                    "parameters": skill.parameters,
                },
            }
            for skill in self._skills.values()
        ]

    def get_skill(self, name: str) -> Skill | None:
        return self._skills.get(name)

    async def execute_skill(self, name: str, **kwargs: Any) -> str:
        skill = self.get_skill(name)
        if not skill:
            return f"Error: Skill '{name}' not found."
        return await skill.execute(**kwargs)


# Global instance
skills_registry = SkillsRegistry()
