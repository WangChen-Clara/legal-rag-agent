from rag_law.ingestion.ecfr_parser import parse_ecfr_xml


def test_parse_section_with_part_context() -> None:
    xml = b"""
    <ROOT>
      <DIV1 TYPE="PART" N="1">
        <HEAD>PART 1 - INVESTMENT SECURITIES</HEAD>
        <DIV2 TYPE="SECTION">
          <HEAD>Sec. 1.1 Authority and scope.</HEAD>
          <P>This part applies to national banks.</P>
        </DIV2>
      </DIV1>
    </ROOT>
    """

    sections = parse_ecfr_xml(xml, title=12, version_date="2025-09-01")

    assert len(sections) == 1
    assert sections[0].part == "1"
    assert sections[0].section == "1.1"
    assert sections[0].text == "Authority and scope.\nThis part applies to national banks."


def test_parse_section_with_letter_suffix_part() -> None:
    xml = """
    <ROOT>
      <DIV1 TYPE="PART" N="261a">
        <HEAD>PART 261a - RULES REGARDING ACCESS TO PERSONAL INFORMATION</HEAD>
        <DIV8 TYPE="SECTION" N="261a.1">
          <HEAD>§ 261a.1 Authority, purpose and scope.</HEAD>
          <P>This section applies to records maintained by the Board.</P>
        </DIV8>
      </DIV1>
    </ROOT>
    """

    sections = parse_ecfr_xml(xml, title=12, version_date="2025-09-01")

    assert len(sections) == 1
    assert sections[0].part == "261a"
    assert sections[0].section == "261a.1"
    assert sections[0].source_url.endswith("/section-261a.1")
