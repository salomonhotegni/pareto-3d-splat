from PIL import Image

from pareto_splat.graphdeco_compat import composite_rgba_image


def test_rgba_image_is_composited_onto_white() -> None:
    image = Image.new("RGBA", (3, 1))
    image.putdata(
        [
            (255, 0, 0, 255),
            (0, 0, 0, 0),
            (255, 0, 0, 128),
        ]
    )

    composited = composite_rgba_image(image, white_background=True)

    assert composited.mode == "RGB"
    assert list(composited.getdata()) == [
        (255, 0, 0),
        (255, 255, 255),
        (255, 127, 127),
    ]


def test_rgba_image_is_composited_onto_black() -> None:
    image = Image.new("RGBA", (1, 1), (255, 0, 0, 128))

    composited = composite_rgba_image(image, white_background=False)

    assert composited.getpixel((0, 0)) == (128, 0, 0)

