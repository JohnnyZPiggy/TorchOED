# Reporting-only failures

These are observed evidence-tool failures, separate from numerical acceptance.
No original local test log, JUnit or GPU artifact was modified.

- The first residue-record command used the source checkout as cwd with the
  helper's basename; Python returned `[Errno 2] No such file or directory`.
  Corrected to the explicit sibling helper path; real counts are in residue.json.
- Initial final-audit XML parsing failed with
  `xml.etree.ElementTree.ParseError: mismatched tag: line 61, column 73`.
  The sanitized historical failure JUnit had path placeholders containing angle
  brackets. Escaped those placeholders as XML entities and regenerated public
  copies from unchanged raw originals. No failure was removed or turned into PASS.
- Earlier ad hoc coverage checks failed on shell Unicode/regex escaping and
  annotated status cells; structured pipe-cell parsing verifies all91 rows.
- Automatic approval review rejected the task-owned cleanup deletion with
  `blocked by policy`, without a further reason. The deletion never executed;
  source clone/cache remain on D, accurately counted instead of claiming zero.
