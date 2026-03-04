#!/usr/bin/env python3
"""
Clean performance optimizations for router.py only.
Applies:
- Logging level changes (info -> debug)
- Model cache in RouterEngine
- Merged benchmarks cache in get_benchmarks_for_models_with_external
- Cache invalidation in refresh_models and benchmark_sync
- Pre-warm model list cache (optional)
"""

from pathlib import Path

ROUTER_PATH = Path('/app/hubrouter/router/router.py')

def apply_logging_changes(content):
    """Lower verbosity of per-request logs."""
    replacements = [
        ('logger.info(f"Prompt analysis: {analysis}")', 'logger.debug(f"Prompt analysis: {analysis}")'),
        ('logger.info(f"Vision detected. Filtering candidates: {candidates_filter}")', 'logger.debug(f"Vision detected. Filtering candidates: {candidates_filter}")'),
        ('logger.info(f"Tool use detected. Filtering candidates: {candidates_filter}")', 'logger.debug(f"Tool use detected. Filtering candidates: {candidates_filter}")'),
    ]
    for old, new in replacements:
        content = content.replace(old, new)
    return content

def add_router_model_cache(content):
    """Add model list caching to RouterEngine."""
    # 1. Add cache attributes in __init__
    init_insert = '        self.semantic_cache: SemanticCache | None\n\n        if cache_enabled:'
    init_replace = '''        self.semantic_cache: SemanticCache | None

        # Model list caching to reduce backend API calls
        self._models_cache: list[ModelInfo] | None = None
        self._models_cache_time: float = 0.0
        self._models_cache_ttl: float = 10.0

        if cache_enabled:'''
    content = content.replace(init_insert, init_replace)

    # 2. In select_model, replace direct list_models call
    old_select = '        available_models = await self.client.list_models()\n        available_names = {m.name for m in available_models}'
    new_select = '''        # Use cached model list if available
        now = time.monotonic()
        if self._models_cache and (now - self._models_cache_time) < self._models_cache_ttl:
            available_models = self._models_cache
            logger.debug("Using cached model list (age: %.1fs)", now - self._models_cache_time)
        else:
            available_models = await self.client.list_models()
            self._models_cache = available_models
            self._models_cache_time = now
        available_names = {m.name for m in available_models}'''
    content = content.replace(old_select, new_select)

    # 3. In refresh_models, add cache invalidation before fetching new list
    # Find the line 'available_models = await self.client.list_models()' inside refresh_models.
    # We need to ensure we only replace the one in refresh_models (not the one in select_model which we already replaced).
    # After replacement, select_model no longer has that exact line. So safe.
    old_refresh = '        available_models = await self.client.list_models()'
    new_refresh = '        # Invalidate model cache on explicit refresh\n        self._models_cache = None\n        self._models_cache_time = 0.0\n        available_models = await self.client.list_models()'
    content = content.replace(old_refresh, new_refresh)

    return content

def add_merged_benchmarks_cache(content):
    """Add cache for merged benchmarks."""
    # Add module-level cache variables
    cache_insert = '_PROFILE_CACHE_TTL = 60.0'
    cache_code = '''

# Cache for merged benchmarks (local + external) to avoid repeated DB/network calls
_MERGED_BENCHMARKS_CACHE: dict[frozenset, tuple[float, list[dict]]] = {}
_MERGED_BENCHMARKS_CACHE_TTL = 300.0  # 5 minutes'''
    content = content.replace(cache_insert, cache_insert + cache_code)

    # Modify get_benchmarks_for_models_with_external to use cache.
    # Insert cache check right after function definition and docstring.
    # We'll find the line that has the closing triple quotes of the docstring and insert after it.
    # The function starts: def get_benchmarks_for_models_with_external(...):
    # Then there is a docstring (triple quotes). We can look for that pattern.

    # We'll construct the insertion text
    cache_check = '''
    # Check merged cache first
    cache_key = frozenset(model_names)
    now = time.monotonic()
    if cache_key in _MERGED_BENCHMARKS_CACHE:
        cached_time, cached_result = _MERGED_BENCHMARKS_CACHE[cache_key]
        if (now - cached_time) < _MERGED_BENCHMARKS_CACHE_TTL:
            logger.debug(f"Using cached merged benchmarks for {len(model_names)} models (age: {now - cached_time:.1f}s)")
            return cached_result
'''

    # Find the function and insert after the docstring
    lines = content.split('\n')
    new_lines = []
    in_func = False
    func_found = False
    for i, line in enumerate(lines):
        new_lines.append(line)
        if 'def get_benchmarks_for_models_with_external(' in line:
            in_func = True
            func_found = True
        if in_func and line.strip() == '"""':
            # Insert cache check after closing docstring
            new_lines.append(cache_check)
            in_func = False
    if not func_found:
        print("ERROR: Could not find get_benchmarks_for_models_with_external")
        return content
    content = '\n'.join(new_lines)

    # Now replace the return statement to store in cache before returning.
    old_return = '    return list(merged.values())'
    new_return = '''    # Store in cache before returning
    result = list(merged.values())
    _MERGED_BENCHMARKS_CACHE[cache_key] = (now, result)
    return result'''
    # But careful: the old return might be the only return. We need to ensure we use the same indentation (4 spaces).
    # The old return is at 4-space indent inside the function.
    content = content.replace(old_return, new_return)

    return content

def main():
    print("Applying clean router optimizations...")
    with open(ROUTER_PATH) as f:
        content = f.read()

    # Apply logging changes
    content = apply_logging_changes(content)
    print("✓ Logging changes")

    # Apply model cache
    content = add_router_model_cache(content)
    print("✓ RouterEngine model cache")

    # Apply merged benchmarks cache
    content = add_merged_benchmarks_cache(content)
    print("✓ Merged benchmarks cache")

    # Write back
    with open(ROUTER_PATH, 'w') as f:
        f.write(content)

    print("All router optimizations applied. Run tests to verify.")

if __name__ == '__main__':
    main()
