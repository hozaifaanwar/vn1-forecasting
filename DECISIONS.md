# Decision Log

Append-only. One entry per material choice. Newest at the bottom.

## D001 — Sales zeros are real observations

**Decision:** Do not interpolate or fill observed sales zeros.

**Reason:** The dataset has a complete weekly grid. A recorded zero is an observation ("nothing sold"), not missing data.

**Revisit if:** stockout, availability, missed-sales, or other censoring information becomes available that allows us to distinguish genuine zero demand from constrained demand.

## D002 — Price forward-fill, per series

**Decision:** Forward-fill price within each Client | Warehouse | Product series only; leave leading NaNs unchanged.

**Reason:** Price appears transaction-derived and sticky — it is observed on sale weeks, and no rows were found where sales > 0 while price was NaN. The last observed price is therefore used as the working estimate for subsequent sale-less weeks.

Leading NaNs are preserved because no historical price was yet observed. Do not backward-fill them, because doing so would use future information.

Preserve a `price_was_missing` flag before forward-filling.

**Revisit if:** a non-transaction price source, promotion file, price history, or other better price information becomes available.

## D003 — Merge validated one-to-one

**Decision:** Sales ↔ price merge uses `validate="one_to_one"`.

**Reason:** Client | Warehouse | Product | date should uniquely identify a row on both sides. Fail loudly on duplicates rather than silently multiplying rows.

**Revisit if:** the underlying data model changes and multiple records per series-date become legitimate.

## D004 — Local validation for a finished competition

**Decision:** Hold out the final 13 weeks of Phase-0 history, generate forecasts for those weeks using only earlier information, and score them using the official competition metric.

**Reason:** The competition is closed, so the local validation setup should reproduce the real forecasting task as faithfully as possible.

**Revisit if:** competition documentation or available historical phases suggest a more faithful rolling or multi-window validation design.

## D005 — Advanced model scope deferred

**Decision:** Deep learning, foundation forecasting models, and advanced ensembling are not required for the first portfolio delivery.

**Reason:** The initial milestone is time-boxed and can demonstrate strong forecasting practice using baselines and global machine-learning models.

These topics are NOT removed from the learning roadmap. Introduce them when conceptually relevant and revisit them for deeper implementation later.

**Revisit if:** the simpler/global ML approaches plateau, the project finishes ahead of schedule, or an advanced method offers a clear learning or performance benefit.
