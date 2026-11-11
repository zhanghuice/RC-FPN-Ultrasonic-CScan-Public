# Public dataset audit summary

The historical local corpus contains 1,130 image/mask records. A SHA-256 byte-content audit found 17 exact-image duplicate pairs: 13 also share identical mask bytes and four have different mask bytes. Ten pairs cross fixed partitions (five Train–Validation and five Train–Test); one further exact image-and-mask pair is duplicated inside Test. Restricted filenames, split membership, and data-derived hashes are deliberately not published.

The official/local 1,126/1,130 difference remains unresolved. Simple duplicate removal does not reproduce the official per-class totals. The authors need an official manifest/version clarification before changing any dataset description.

Frozen-checkpoint diagnostic exclusion changed absolute metrics, confirming that the historical estimates are affected. Because validation leakage influenced checkpoint selection, only a group-disjoint split followed by retraining can establish corrected performance.
