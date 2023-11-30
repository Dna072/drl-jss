from custom_environment.recipe import Recipe


def create_recipe(
    factory_id: str, process_time: float, process_id: int, recipe_type: str
):
    """
    Factory function for creating a Recipe object
    :param factory_id: the ID given to the recipe by the factory for identification
    :param process_time: the estimated time for the recipe to be completed
    :param process_id: the ID of the Recipe in respect to RL algorithm
    :param recipe_type: the type of the recipe as given by the factory
    :return: Recipe object
    """
    return Recipe(
        factory_id=factory_id,
        process_time=process_time,
        process_id=process_id,
        recipe_type=recipe_type,
    )


if __name__ == "__main__":
    recipes: list[Recipe] = [
        create_recipe(
            factory_id="R1_ID", process_time=2.0, process_id=0, recipe_type="R1"
        ),
        create_recipe(
            factory_id="R2_ID", process_time=10.0, process_id=1, recipe_type="R2"
        ),
    ]

    print("Recipes:")
    for recipe in recipes:
        print(recipe)
        print("-------")
