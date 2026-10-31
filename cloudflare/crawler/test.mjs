// Same fixture as tests/test_crawler.py — keeps the JS port honest.
import assert from "node:assert/strict";
import { parseHtml } from "./src/index.js";

const FIXTURE = `
<table><tbody>
  <tr><th>Ky</th><th>Ngay</th><th>Ket qua</th></tr>
  <tr><td>18/07/2026</td><td>01373</td><td>
    <span>22</span><span>41</span><span>45</span><span>48</span>
    <span>54</span><span>55</span><span>|</span><span>16</span></td></tr>
  <tr><td>16/07/2026</td><td>01372</td><td>
    <span>19</span><span>20</span><span>33</span><span>45</span>
    <span>48</span><span>53</span><span>|</span><span>21</span></td></tr>
</tbody></table>`;

const rows = parseHtml(FIXTURE);
assert.equal(rows.length, 2);
assert.equal(rows[0].date, "2026-07-18");
assert.equal(rows[0].id, "01373");
assert.deepEqual(rows[0].result, [22, 41, 45, 48, 54, 55, 16]);
assert.deepEqual(parseHtml(""), []);
assert.deepEqual(parseHtml("<table></table>"), []);
console.log("parser ok");
