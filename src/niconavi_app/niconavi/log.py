import logging
from logging import FileHandler, Formatter, StreamHandler, getLogger


def set_logger() -> None:
    logger = getLogger("niconavi")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        return

    handler_format = Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    stream_handler = StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(handler_format)

    file_handler = FileHandler("niconavi.log", "a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(handler_format)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
