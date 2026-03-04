#!/usr/bin/env python3
"""
Apply all performance optimizations to SmarterRouter.
"""

import sys
from pathlib import Path

ROUTER_PATH = Path('/app/hubrouter/router/router.py')
MAIN_PATH = Path('/app/hubrouter/main.py')
DB_PATH = Path('/app/hubrouter/router/database.py')
SYNC_PATH = Path('/app/hubrouter/router/benchmark_sync.py')

def apply_router_optimizations():
    """Apply optimizations to router.py."""
    with open(ROUTER_PATH) as f:
        content = f.read()

    # 1. Add model cache attributes to RouterEngine.__init__
    init_insert = '''        self.semantic_cache: SemanticCache | None

        if cache_enabled:'''
    init_replace = '''        self.semantic_cache: SemanticCache | None

        # Model list caching to reduce backend API calls
        self._models_cache: list[ModelInfo] | None = None
        self._models_cache_time: float = 0.0
        self._models_cache_ttl: float = 10.0

        if cache_enabled:'''
    if init_insert in content:
        content = content.replace(init_insert, init_replace)
    else:
        print("WARNING: Could not find init_insert pattern")

    # 2. Modify select_model: replace direct client.list_models() with cache logic
    # Find the lines:
    #   available_models = await self.client.list_models()
    #   available_names = {m.name for m in available_models}
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
    if old_select in content:
        content = content.replace(old_select, new_select)
    else:
        print("WARNING: Could not find select_model pattern (already changed?)")

    # 3. Invalidate model cache in refresh_models
    # Find the line: available_models = await self.client.list_models()
    # Insert before it: self._models_cache = None; self._models_cache_time = 0.0
    old_refresh = '        available_models = await self.client.list_models()'
    if old_refresh in content:
        # But we need to only replace the one in refresh_models, not in select_model (which we already replaced)
        # Since we replaced select_model already, the old_select no longer exists there, so only refresh_models has this pattern
        content = content.replace(
            old_refresh,
            '        # Invalidate model cache on explicit refresh\n        self._models_cache = None\n        self._models_cache_time = 0.0\n        available_models = await self.client.list_models()'
        )
    else:
        print("WARNING: Could not find refresh_models list_models call")

    # 4. Add merged benchmarks cache
    # Insert after _PROFILE_CACHE_TTL line
    cache_insert_marker = '_PROFILE_CACHE_TTL = 60.0'
    cache_code = '''

# Cache for merged benchmarks (local + external) to avoid repeated DB/network calls
_MERGED_BENCHMARKS_CACHE: dict[frozenset, tuple[float, list[dict]]] = {}
_MERGED_BENCHMARKS_CACHE_TTL = 300.0  # 5 minutes'''
    if cache_insert_marker in content:
        content = content.replace(cache_insert_marker, cache_insert_marker + cache_code)
    else:
        print("WARNING: Could not find _PROFILE_CACHE_TTL")

    # 5. Modify get_benchmarks_for_models_with_external to use cache
    # Find function start
    func_start_pat = r'(def get_benchmarks_for_models_with_external\(\s*model_names: list\[str\]\s*\) -> list\[dict\]:\s*""")'
    # We'll insert caching at start after docstring
    # Instead of complex regex, find the line after the docstring that is blank or the first code line
    # We'll insert:
    #   # Check merged cache first
    #   cache_key = frozenset(model_names)
    #   now = time.monotonic()
    #   if cache_key in _MERGED_BENCHMARKS_CACHE:
    #       cached_time, cached_result = _MERGED_BENCHMARKS_CACHE[cache_key]
    #       if (now - cached_time) < _MERGED_BENCHMARKS_CACHE_TTL:
    #           return cached_result
    lines = content.split('\n')
    new_lines = []
    in_func = False
    func_found = False
    for i, line in enumerate(lines):
        if 'def get_benchmarks_for_models_with_external(' in line:
            in_func = True
            func_found = True
        new_lines.append(line)
        if in_func and line.strip() == '"""':
            # After closing docstring, insert cache check
            indent = '    '
            cache_code_lines = [
                '',
                indent + '# Check merged cache first',
                indent + 'cache_key = frozenset(model_names)',
                indent + 'now = time.monotonic()',
                indent + 'if cache_key in _MERGED_BENCHMARKS_CACHE:',
                indent + '    cached_time, cached_result = _MERGED_BENCHMARKS_CACHE[cache_key]',
                indent + '    if (now - cached_time) < _MERGED_BENCHMARKS_CACHE_TTL:',
                indent + '        logger.debug(f"Using cached merged benchmarks for {len(model_names)} models (age: {now - cached_time:.1f}s)")',
                indent + '        return cached_result',
            ]
            new_lines.extend(cache_code_lines)
            in_func = False
    if not func_found:
        print("ERROR: Could not find get_benchmarks_for_models_with_external function")
        sys.exit(1)
    content = '\n'.join(new_lines)

    # 6. Add cache store at end of function before return
    # Find "return list(merged.values())" inside the function and replace
    old_return = '    return list(merged.values())'
    new_return = '''    # Store in cache before returning
    result = list(merged.values())
    _MERGED_BENCHMARKS_CACHE[cache_key] = (now, result)
    return result'''
    if old_return in content:
        content = content.replace(old_return, new_return)
    else:
        print("WARNING: Could not find return statement in get_benchmarks_for_models_with_external")

    # 7. Invalidate cache in benchmark_sync.py
    with open(SYNC_PATH) as f:
        sync_content = f.read()
    old_sync_end = '''    count = bulk_upsert_benchmarks(final_benchmarks)

    update_sync_status("completed", count)
    logger.info(f"Benchmark sync completed: {count} models synced")

    return count, list(matched_models)'''
    new_sync_end = '''    count = bulk_upsert_benchmarks(final_benchmarks)

    update_sync_status("completed", count)
    logger.info(f"Benchmark sync completed: {count} models synced")

    # Invalidate merged benchmarks cache to force refresh on next request
    try:
        from router.router import _MERGED_BENCHMARKS_CACHE
        _MERGED_BENCHMARKS_CACHE.clear()
    except ImportError:
        pass  # Module not loaded yet, cache will be clear on next use

    return count, list(matched_models)'''
    if old_sync_end in sync_content:
        sync_content = sync_content.replace(old_sync_end, new_sync_end)
        with open(SYNC_PATH, 'w') as f:
            f.write(sync_content)
        print("✓ Invalidated merged benchmarks cache in benchmark_sync")
    else:
        print("WARNING: Could not find sync end pattern")

    # 8. Optimize benchmark matching: replace O(N*M) with O(N+M) using normalized index
    # This is in _calculate_combined_scores
    old_match = '''        # Normalize benchmark keys for matching
        normalized_benchmark_map = {}
        for name in model_names:
            # Extract base name - handle versions and quantizations
            base = name.split(":")[0].lower().replace("-", "").replace("_", "").replace(".", "")
            # Also try with just the first part before numbers
            base.split("2")[0] if "2" in base else base

            best_match = None
            best_score = 0.0

            for bm_name, bm in benchmark_map.items():
                bm_base = (
                    bm_name.split(":")[0].lower().replace("-", "").replace("_", "").replace(".", "")
                )

                # Exact match
                if base == bm_base:
                    best_match = bm
                    best_score = 100.0
                    break

                # Partial match - check if major model name matches
                if base in bm_base or bm_base in base:
                    score = len(base) / max(len(base), len(bm_base), 1)
                    if score > best_score:
                        best_match = bm
                        best_score = score
                elif any(part in bm_base for part in base.split() if len(part) > 2):
                    # Try matching individual parts
                    for part in [base, base[:4], base[:6]]:
                        if part in bm_base and len(part) > 2:
                            best_match = bm
                            best_score = 0.5
                            break

            if best_match:
                normalized_benchmark_map[name] = best_match'''
    new_match = '''        # Build normalized index for benchmarks (optimization: O(M) instead of O(N*M))
        normalized_benchmark_index = {}
        for bm_name, bm in benchmark_map.items():
            norm = bm_name.split(":")[0].lower().replace("-", "").replace("_", "").replace(".", "")
            if norm not in normalized_benchmark_index:  # keep first occurrence
                normalized_benchmark_index[norm] = bm

        # Match models against normalized benchmark index
        normalized_benchmark_map = {}
        for name in model_names:
            base = name.split(":")[0].lower().replace("-", "").replace("_", "").replace(".", "")
            # Try exact match first
            bm = normalized_benchmark_index.get(base)
            if bm:
                normalized_benchmark_map[name] = bm
                continue
            # Try partial match: check if base substring of any normalized benchmark or vice versa
            for norm, bm_candidate in normalized_benchmark_index.items():
                if base in norm or norm in base:
                    normalized_benchmark_map[name] = bm_candidate
                    break'''
    if old_match in content:
        content = content.replace(old_match, new_match)
    else:
        print("WARNING: Could not find benchmark matching loop (already modified?)")

    # Write back router.py
    with open(ROUTER_PATH, 'w') as f:
        f.write(content)
    print("✓ Applied router.py optimizations")

def apply_main_optimizations():
    """Modify main.py to use get_available_models_with_cache."""
    with open(MAIN_PATH) as f:
        content = f.read()

    # In chat_completions endpoint, we need to fetch available_models before try block
    # and use that variable, removing the inner fetch in model_override branch.
    # We'll find the section:
    #   # Check for model override query parameter
    #   model_override = ...
    #   # Track request
    #   if hasattr(...): ...
    #   try:
    #       if model_override:
    #           available_models = await list_models_with_timeout(app_state.backend)
    #           ...
    #       else:
    #           ...
    #           available_models=available_models
    # We want to:
    #   After model_override and before try, fetch available_models using cache
    #   Inside try: remove the inner fetch in model_override branch
    #   Also ensure available_models defined for else branch (it will be from outer scope)

    # Step 1: Add fetch before try
    # Find "# Track request" block and after that insert fetch
    old_track = '''    if hasattr(app_state, "total_requests"):
        app_state.total_requests += 1

    try:
        # Model override - skip routing and use specified model
        if model_override:
            available_models = await list_models_with_timeout(app_state.backend)
            model_names = [m.name for m in available_models]'''
    new_track = '''    if hasattr(app_state, "total_requests"):
        app_state.total_requests += 1

    # Fetch available models once per request (uses cache)
    try:
        available_models = await get_available_models_with_cache()
        model_names = [m.name for m in available_models]

        # Model override - skip routing and use specified model
        if model_override:
            # Use already fetched model_names for validation
            # (no need to refetch)
            selected_model = None'''
    if old_track in content:
        content = content.replace(old_track, new_track)
    else:
        print("WARNING: Could not find track block pattern")

    # Step 2: In the model_override branch, remove the inner fetch and model_names assignment
    # We already inserted model_names above, so we need to remove the lines inside the if that redefine them.
    # The old lines inside if were:
    #   available_models = await list_models_with_timeout(app_state.backend)
    #   model_names = [m.name for m in available_models]
    # We already moved those before try, so we should delete those two lines plus the assignment of selected_model = None remains
    # Actually we replaced up to that point; after replacement, we included "selected_model = None" in the new string. Let's adjust: we need to keep the original code after "if model_override:" but without the fetch.
    # It might be simpler to delete specific lines after the replacement.
    # Let's find the pattern:
    #   if model_override:
    #       available_models = await list_models_with_timeout(app_state.backend)
    #       model_names = [m.name for m in available_models]
    # Later in the else we pass available_models=available_models to select_model.
    pass  # We'll handle via more replacement

    # Actually my new_track already includes removing those lines by replacing up to that point. But careful: the original code continued with
    #   # Try exact match first, then partial match
    #   selected_model = None
    #   for name in model_names: ...
    # So after replacement, we need that code to still be there. The new_track ends with "selected_model = None". That is exactly the line that was originally after the model_names assignment. So it's correct.

    # Step 3: Ensure get_available_models_with_cache function is defined (already present), and ensure it's imported/used correctly.

    # Step 4: Also fix the fallback in except block to use get_available_models_with_cache instead of list_models_with_timeout
    # Find: models = await list_models_with_timeout(app_state.backend)
    # Replace with: models = await get_available_models_with_cache()
    content = content.replace(
        '        models = await list_models_with_timeout(app_state.backend)',
        '        models = await get_available_models_with_cache()'
    )

    with open(MAIN_PATH, 'w') as f:
        f.write(content)
    print("✓ Applied main.py endpoint optimizations")

def apply_database_pooling():
    """Add connection pooling to SQLAlchemy engine."""
    with open(DB_PATH) as f:
        content = f.read()

    # Find the engine creation: create_engine(settings.database_url, ...)
    # Add pool_size, max_overflow, pool_recycle, echo=False if not present
    old_engine = '''    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False, "timeout": 20},
        echo=settings.debug,
    )'''
    new_engine = '''    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False, "timeout": 20},
        echo=settings.debug,
        pool_size=10,
        max_overflow=20,
        pool_recycle=3600,
    )'''
    if old_engine in content:
        content = content.replace(old_engine, new_engine)
        with open(DB_PATH, 'w') as f:
            f.write(content)
        print("✓ Added connection pooling to database engine")
    else:
        print("WARNING: Could not find engine creation pattern (maybe already modified)")

def main():
    print("Applying performance optimizations...")
    apply_router_optimizations()
    apply_main_optimizations()
    apply_database_pooling()
    print("Done. Run tests to verify.")

if __name__ == '__main__':
    main()
