# Review evidence and current locked export

The existing author-confirmed workbook is preserved in the local submission
package. The GitHub repository includes its two correction records and the
current CSV export. The active transcripts_reviewed.jsonl includes both additions.

locked_review.csv is the current 150-record export: original text, locked gold,
clinical annotations, reference text and the two author-confirmed correction
markers. It is linked to the final JSONL by the offline verifier. The workbook
and CSV are review evidence, not an assertion that every row was approved.

gold_corrections.csv records only SIM059 PERSON Dana and SIM082 SESSION_PATTERN
every other Tuesday. The before/after metric CSV and recalculation notes are
historical evidence of that rescore. No further annotation change was made
during package synchronization. Current authoritative results are under
data/results/tables/ and outputs_for_manuscript/.
