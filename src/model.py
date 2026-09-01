import base64
import time
from pathlib import Path

from openai import OpenAI

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    MODEL_NAME,
)


client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)


def encode_image(image_path: str) -> str:
    """Convert an image to base64."""

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with open(image_path, "rb") as image_file:
        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


def get_image_mime_type(image_path: str) -> str:
    """Return the MIME type for the image."""

    extension = Path(image_path).suffix.lower()

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }

    if extension not in mime_types:
        raise ValueError(
            f"Unsupported image format: {extension}"
        )

    return mime_types[extension]


def generate_caption(
    image_path: str,
    prompt: str,
    temperature: float,
    max_retries: int = 3,
):
    """
    Generate a caption with controlled retries.

    Returns:
        {
            "output": str,
            "attempts": int
        }
    """

    encoded_image = encode_image(image_path)

    mime_type = get_image_mime_type(image_path)

    last_error = None

    for attempt in range(1, max_retries + 1):

        try:

            response = client.chat.completions.create(
                model=MODEL_NAME,

                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        f"data:{mime_type};base64,"
                                        f"{encoded_image}"
                                    )
                                },
                            },
                        ],
                    }
                ],

                temperature=temperature,
            )
            # Validate response

            if response is None:
                raise RuntimeError(
                    "Model returned None response."
                )

            if not response.choices:
                raise RuntimeError(
                    "Model response contains no choices."
                )

            message = response.choices[0].message

            if message is None:
                raise RuntimeError(
                    "Model response contains no message."
                )

            content = message.content

            if content is None:
                raise RuntimeError(
                    "Model response contains empty content."
                )

            content = content.strip()

            if not content:
                raise RuntimeError(
                    "Model returned empty caption."
                )


            # SUCCESS
            return {
                "output": content,
                "attempts": attempt,
            }

        except Exception as error:

            last_error = error

            print(
                f"\n    Attempt "
                f"{attempt}/{max_retries} failed: "
                f"{error}"
            )

            if attempt < max_retries:

                # Exponential backoff
                wait_time = 2 ** attempt

                print(
                    f"    Retrying in "
                    f"{wait_time} seconds..."
                )

                time.sleep(wait_time)

    # ALL ATTEMPTS FAILED

    raise RuntimeError(
        f"Generation failed after "
        f"{max_retries} attempts: "
        f"{last_error}"
    )