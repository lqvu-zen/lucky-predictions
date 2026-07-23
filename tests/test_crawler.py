"""The HTML parser is the fragile part most likely to break if the site
changes — pin its behaviour against a fixture."""
from crawler import _parse_html

FIXTURE = """
<table><tbody>
  <tr><th>Ky</th><th>Ngay</th><th>Ket qua</th></tr>
  <tr><td>18/07/2026</td><td>01373</td><td>
    <span>22</span><span>41</span><span>45</span><span>48</span>
    <span>54</span><span>55</span><span>|</span><span>16</span></td></tr>
  <tr><td>16/07/2026</td><td>01372</td><td>
    <span>19</span><span>20</span><span>33</span><span>45</span>
    <span>48</span><span>53</span><span>|</span><span>21</span></td></tr>
</tbody></table>
"""


def test_parses_rows_and_converts_dates():
    rows = _parse_html(FIXTURE)
    assert len(rows) == 2
    r = rows[0]
    assert r["date"] == "2026-07-18"          # dd/mm/yyyy -> ISO
    assert r["id"] == "01373"
    assert r["result"] == [22, 41, 45, 48, 54, 55, 16]  # 6 main + bonus, '|' dropped


def test_empty_html_is_safe():
    assert _parse_html("") == []
    assert _parse_html("<table></table>") == []
