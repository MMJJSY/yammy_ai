def score_recipe(
    user_ingredients: list[str],
    recipe_ingredients: list[str]
) -> float:
    user_set = set(user_ingredients)
    recipe_set = set(recipe_ingredients)

    matched = user_set & recipe_set
    missing = recipe_set - user_set

    if not recipe_set:
        return 0.0

    match_count = len(matched)
    match_ratio = match_count / len(recipe_set)

    score = (
        match_count * 3
        + match_ratio * 5
        - len(missing) * 1
    )

    return round(score, 2)