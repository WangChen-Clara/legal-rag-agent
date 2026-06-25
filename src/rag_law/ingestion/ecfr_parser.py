from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


SECTION_NUMBER = re.compile(
    r"^[^\d]*([0-9]+[A-Za-z]?(?:\.[0-9A-Za-z-]+)*)", re.I
)
PART_NUMBER = re.compile(r"(?:PART\s+)?([0-9]+[A-Za-z]?)", re.I)
SECTION_PREFIX = re.compile(
    r"^[^\d]*[0-9]+[A-Za-z]?(?:\.[0-9A-Za-z-]+)*\s*[-‐‑‒–—]?\s*",
    re.I,
)
BODY_TAGS = {"P", "NOTE", "EXAMPLE", "XREF", "FP"}


@dataclass(frozen=True)
class ECFRSection:
    title: int
    part: str
    section: str
    heading: str
    text: str
    version_date: str
    source_url: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1].upper()


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return re.sub(r"\s+", " ", " ".join(element.itertext())).strip()


def _first_descendant(element: ET.Element, tag: str) -> ET.Element | None:
    wanted = tag.upper()
    return next((child for child in element.iter() if _tag(child) == wanted), None)


def _division_type(element: ET.Element) -> str:
    if not _tag(element).startswith("DIV"):
        return ""
    return (element.get("TYPE") or "").upper()


def _part_number(element: ET.Element, fallback: str) -> str:
    for value in (element.get("N"), _text(_first_descendant(element, "HEAD"))):
        if value:
            match = PART_NUMBER.search(value)
            if match:
                return match.group(1)
    return fallback


def _legacy_compatible_text(section_node: ET.Element, heading: str) -> str:
    chunks: list[str] = []
    clean_heading = SECTION_PREFIX.sub("", heading).strip()
    if clean_heading:
        chunks.append(clean_heading)
    for element in section_node.iter():
        if _tag(element) in BODY_TAGS:
            value = _text(element)
            if value:
                chunks.append(value)
    return "\n".join(chunks).strip()


def parse_ecfr_xml(
    xml: str | bytes,
    *,
    title: int,
    version_date: str,
    requested_part: str = "",
) -> list[ECFRSection]:
    root = ET.fromstring(xml)
    sections: list[ECFRSection] = []

    def walk(element: ET.Element, current_part: str) -> None:
        division_type = _division_type(element)
        if division_type == "PART":
            current_part = _part_number(element, current_part or requested_part)
        if division_type == "SECTION":
            heading = _text(_first_descendant(element, "HEAD"))
            match = SECTION_NUMBER.search(heading)
            if match:
                section_number = match.group(1)
                part = current_part or requested_part or section_number.split(".", 1)[0]
                sections.append(
                    ECFRSection(
                        title=title,
                        part=part,
                        section=section_number,
                        heading=heading,
                        text=_legacy_compatible_text(element, heading),
                        version_date=version_date,
                        source_url=(
                            f"https://www.ecfr.gov/on/{version_date}/"
                            f"title-{title}/section-{section_number}"
                        ),
                    )
                )
            return
        for child in element:
            walk(child, current_part)

    walk(root, requested_part)
    return sections


def parse_ecfr_file(
    path: str | Path, *, title: int, version_date: str, requested_part: str = ""
) -> list[ECFRSection]:
    return parse_ecfr_xml(
        Path(path).read_bytes(),
        title=title,
        version_date=version_date,
        requested_part=requested_part,
    )
