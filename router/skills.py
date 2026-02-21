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
    async def execute(self, **kwargs: Any) -> str:
        """Perform a web search using DuckDuckGo API."""
        query = kwargs.get("query", "")
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
    """Safely evaluate mathematical expressions with proper security."""
    
    # Maximum result magnitude to prevent DoS
    MAX_RESULT = 1e15
    
    # Only allow these operators (no exponentiation to prevent DoS)
    ALLOWED_OPERATORS = {"+", "-", "*", "/"}

    async def execute(self, **kwargs: Any) -> str:
        """Safely evaluate a mathematical expression using ast parsing."""
        expression = kwargs.get("expression", "")
        
        if not expression:
            return "Error: expression parameter is required."
        
        # Security: limit expression length
        if len(expression) > 100:
            return "Error: expression too long (max 100 characters)."
        
        try:
            import ast
            import operator
            
            # Define allowed operations
            operators = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.USub: operator.neg,  # Unary minus
            }
            
            def safe_eval(node):
                """Recursively evaluate AST node with safety checks."""
                if isinstance(node, ast.Constant):
                    if isinstance(node.value, (int, float)):
                        return float(node.value)
                    raise ValueError("Only numeric values allowed")
                
                elif isinstance(node, ast.BinOp):
                    if type(node.op) not in operators:
                        raise ValueError(f"Operator not allowed: {type(node.op).__name__}")
                    
                    left = safe_eval(node.left)
                    right = safe_eval(node.right)
                    result = operators[type(node.op)](left, right)
                    
                    # Prevent overflow/DoS
                    if abs(result) > self.MAX_RESULT:
                        raise ValueError("Result too large")
                    return result
                
                elif isinstance(node, ast.UnaryOp):
                    if type(node.op) not in operators:
                        raise ValueError(f"Unary operator not allowed")
                    operand = safe_eval(node.operand)
                    return operators[type(node.op)](operand)
                
                else:
                    raise ValueError(f"Expression type not allowed: {type(node).__name__}")
            
            # Parse and validate expression
            tree = ast.parse(expression, mode="eval")
            result = safe_eval(tree.body)
            
            # Format result nicely
            if result == int(result):
                return str(int(result))
            return str(round(result, 10))
            
        except ValueError as e:
            return f"Error: {e}"
        except SyntaxError:
            return "Error: Invalid expression syntax."
        except ZeroDivisionError:
            return "Error: Division by zero."
        except Exception as e:
            return f"Error evaluating expression: {type(e).__name__}"


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
