import os
from kg_gen import KGGen


def main():
    print("=== KGGen test ===")

    # Sanity checks
    print("OPENAI_API_KEY present:", bool(os.getenv("OPENAI_API_KEY")))

    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set"
    key_to_test = os.getenv("OPENAI_API_KEY")
    kg = KGGen(
        api_base='https://ai-research-proxy.azurewebsites.net',
        api_key=key_to_test,
        model="azure/nf-gpt-4o-mini",
        temperature=1.0,
        max_tokens=4096
    )

    input_texts = [
        ####### sample input passage 1:
        "Excerpt from the Wikipedia page describing Dan Brouillette: '''Danny Ray Brouillette''' "
        "( born August 18, 1962) is an American government official and businessman who has served "
        "as the United States Secretary of Energy since December 4, 2019. He previously served as "
        "the Deputy Secretary of Energy from August 2017 to December 2019. Brouillette left office"
        " on January 20, 2021 when Joe Biden was sworn in as President.",
        ####### sample input passage 2:
        "Excerpt from the Wikipedia page describing 2021 NFL draft: == 2020 Resolution JC-2A picks "
        "== * San Francisco received third-round selections in 2021, 2022, and 2023, when their "
        "defensive coordinator Robert Saleh was hired by the New York Jets as head coach, and"
        " their vice president of player personnel Martin Mayhew was hired by Washington as general"
        " manager. * The Los Angeles Rams received third-round selections in 2021 and 2022 when "
        "their college scouting director Brad Holmes was hired by Detroit as general manager."
        " * New Orleans received third-round selections in 2021 and 2022 when their director of"
        " pro scouting Terry Fontenot was hired by Atlanta as general manager. * Baltimore received"
        " third-round selections in 2021 and 2022 when their assistant head coach and passing game"
        " coordinator David Culley was hired by Houston as head coach.",
        ####### sample input passage 3:
        "Excerpt from the Wikipedia page describing White House Chief of Staff: In the administration"
        " of President Joe Biden, the current chief of staff is Ron A. Klain, who succeeded "
        "Mark Meadows on January 20, 2021 after the Trump administration failed to secure a"
        " second term in The White House. The position is widely recognized as one of great power and"
        " influence, owing to daily contact with the president and control of the White House Office."
    ]
    for text in input_texts:
        print(f"processing text: {text}")
        graph = kg.generate(
            input_data=text,
        )

        print("\nReturned graph:")
        print(graph)

        if hasattr(graph, "relations"):
            print("\nRelations:")
            for r in graph.relations:
                print("  ", r)
        else:
            print("No relations attribute found")

    print("\n=== SUCCESS ===")


if __name__ == "__main__":
    main()
