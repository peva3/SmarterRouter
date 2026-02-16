from dataclasses import dataclass
from typing import Any

@dataclass
class Skill:
    name: str
    description: str
    parameters: dict[str, Any]

class SkillsRegistry:
    def __init__(self):
        self._skills: dict[str, Skill] = {}
        # Register some default skills
        self.register(Skill(
            name="web_search",
            description="Search the web for current information.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        ))
        self.register(Skill(
            name="calculator",
            description="Perform mathematical calculations.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Mathematical expression to evaluate (e.g., '2 + 2')"}
                },
                "required": ["expression"]
            }
        ))

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def list_skills(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": s.name,
                    "description": s.description,
                    "parameters": s.parameters
                }
            }
            for s in self._skills.values()
        ]

    def get_skill(self, name: str) -> Skill | None:
        return self._skills.get(name)

# Global instance
skills_registry = SkillsRegistry()
