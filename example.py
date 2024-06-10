from typing import List

from punctuators.models import SBDModelONNX
import click


@click.command()
@click.option("--model-name", default="sbd_multi_lang", help="Name of the model to use")
@click.argument(
    "input_files",
    nargs=-1,
    type=click.File("r", encoding="utf-8"),
    required=False,
)
def main(model_name, input_files) -> None:

    # Get stdin, stderr and stdout
    stdout = click.get_text_stream("stdout")

    # Instantiate this model
    m = SBDModelONNX.from_pretrained(model_name)

    # Read texts from stdin
    input_texts: List[str] = []

    for input_file in input_files:
        for line in input_file:
            input_texts.append(line.strip())

    # Run inference
    results: List[List[str]] = m.infer(input_texts)
    results_flat = [text for texts in results for text in texts]

    # Print each input and it's segmented outputs

    for text in results_flat:
        click.echo(text, file=stdout)


if __name__ == "__main__":
    main()
