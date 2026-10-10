"""2026 dollar amounts from the handbook, as a table a model can look up instead of recall.

A model asked "what is the Part B premium?" tends to answer from memory, and remembered Medicare
amounts are usually last year's. This table holds the single-valued 2026 amounts printed on pages
23, 27, 29, 30, 53, 76, 83, 84 and 88, each with the sentence it was copied from, so an answer
that uses one can quote the page it came from. It is not every amount the handbook prints: page
22's Part A premium, for one, gives two amounts without saying who pays which, and stays in the
retrieved passages.

Amounts are `Decimal`, never `float`: $202.90 stays $202.90 through any arithmetic. There is no
Part D deductible here on purpose: the handbook says what a deductible is but never prints the
largest one a plan may charge, so the honest answer to that question is that the handbook does not
say. A test checks every `source_text` against the page text of the pinned handbook.
"""

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class CostFigure:
    """One dollar amount from the handbook, with what it is for and where it is printed.

    `source_text` is the handbook's own sentence, copied from the page with its whitespace
    collapsed, and it contains `amount` exactly as the page prints it.
    """

    key: str
    label: str
    amount: Decimal
    unit: str
    applies_to: str
    page: int
    source_text: str


_FIGURES = (
    CostFigure(
        key="part_b_standard_premium",
        label="Part B standard premium",
        amount=Decimal("202.90"),
        unit="per month",
        applies_to=(
            "The monthly Part B premium most people with Part B pay in 2026; people with "
            "higher incomes may also pay an extra amount called IRMAA."
        ),
        page=23,
        source_text=(
            "The standard Part B premium amount in 2026 is $202.90. Most people pay the "
            "standard Part B premium amount every month."
        ),
    ),
    CostFigure(
        key="part_b_deductible",
        label="Part B deductible",
        amount=Decimal("283"),
        unit="per year",
        applies_to=(
            "What someone with Original Medicare pays each year for Part B-covered services "
            "before Medicare starts to pay its share."
        ),
        page=30,
        source_text=(
            "Under Original Medicare, if the Part B deductible ($283 in 2026) applies, you must "
            "pay all costs (up to the Medicare-approved amount) until you meet the yearly Part B "
            "deductible."
        ),
    ),
    CostFigure(
        key="part_a_deductible",
        label="Part A deductible",
        amount=Decimal("1736"),
        unit="per benefit period",
        applies_to=(
            "What someone pays under Part A each time a new benefit period starts, before "
            "Medicare starts to pay, for inpatient hospital care and, unless already paid in the "
            "same benefit period, skilled nursing facility care (page 29)."
        ),
        page=27,
        source_text=(
            "Each time you start a new benefit period, you must pay $1,736 (in 2026) before "
            "Medicare starts to pay. During a benefit period, what you pay depends on how long "
            "you’re getting inpatient hospital care: • Days 1–60: After you pay the $1,736 "
            "Part A deductible, you pay $0 each day."
        ),
    ),
    CostFigure(
        key="part_a_coinsurance_days_61_90",
        label="Part A hospital coinsurance, days 61–90",
        amount=Decimal("434"),
        unit="per day",
        applies_to=(
            "What someone pays for each day of inpatient hospital care from day 61 to day 90 "
            "of a benefit period."
        ),
        page=27,
        source_text="Days 61–90: You pay $434 each day.",
    ),
    CostFigure(
        key="part_a_coinsurance_lifetime_reserve_days",
        label="Part A hospital coinsurance, lifetime reserve days",
        amount=Decimal("868"),
        unit="per day",
        applies_to=(
            "What someone pays for each day of inpatient hospital care from day 91 to day 150 "
            "of a benefit period, while using their 60 lifetime reserve days."
        ),
        page=27,
        source_text=(
            "Days 91–150: You pay $868 each day while using your 60 lifetime reserve days. "
            "These are additional days that Medicare will pay for when you’re in a hospital "
            "for more than 90 days. You have a total of 60 reserve days that can be used once "
            "during your lifetime."
        ),
    ),
    CostFigure(
        key="snf_coinsurance_days_21_100",
        label="Skilled nursing facility coinsurance, days 21–100",
        amount=Decimal("217"),
        unit="per day",
        applies_to=(
            "What someone pays for each day of skilled nursing facility care from day 21 to "
            "day 100 of a benefit period."
        ),
        page=29,
        source_text="Days 21–100: You pay $217 each day.",
    ),
    CostFigure(
        key="part_d_out_of_pocket_cap",
        label="Part D out-of-pocket cap",
        amount=Decimal("2100"),
        unit="per year",
        applies_to=(
            "The most someone pays out of pocket in 2026 for drugs their Medicare drug plan "
            "(Part D) covers; after that, covered Part D drugs cost nothing for the rest of "
            "the year."
        ),
        page=83,
        source_text=(
            "Your yearly out-of-pocket drug costs for drugs covered by your plan are capped at "
            "$2,100 in 2026."
        ),
    ),
    CostFigure(
        key="insulin_monthly_cap",
        label="Part D insulin cap",
        amount=Decimal("35"),
        unit="per one-month supply",
        applies_to=(
            "The most a Medicare drug plan can charge for a one-month supply of each Part "
            "D-covered insulin, with no deductible for insulin; page 38 states the same cap."
        ),
        page=88,
        source_text=(
            "Plans can’t charge you more than $35 for a one-month supply of each Part "
            "D-covered insulin you take, and you don’t have to pay a deductible for insulin."
        ),
    ),
    CostFigure(
        key="part_d_national_base_beneficiary_premium",
        label="Part D national base beneficiary premium",
        amount=Decimal("38.99"),
        unit="in 2026",
        applies_to=(
            "The base the Part D late enrollment penalty is computed from (1% of it for each "
            "full month without creditable drug coverage); it is not a premium anyone pays as "
            "such, and page 85 works the same example with it."
        ),
        page=84,
        source_text=(
            "Currently, the late enrollment penalty is calculated by multiplying 1% of the "
            "“national base beneficiary premium” ($38.99 in 2026) by the number of full, "
            "uncovered months that you were eligible but didn’t have Medicare drug coverage "
            "(Part D) and went without other creditable prescription drug coverage. The final "
            "amount is rounded to the nearest $.10 and added to your monthly premium."
        ),
    ),
    CostFigure(
        key="medigap_high_deductible_plan_deductible",
        label="Medigap high-deductible Plan F and G deductible",
        amount=Decimal("2950"),
        unit="in 2026",
        applies_to=(
            "What someone with the high-deductible version of Medigap Plan F or Plan G, offered "
            "in some states, pays in Medicare-covered costs before the policy pays anything."
        ),
        page=76,
        source_text=(
            "Plans F and G also offer a high-deductible plan in some states. You must pay "
            "Medicare-covered costs (coinsurance, copayments, and deductibles) up to the "
            "deductible amount of $2,950 in 2026 before your policy pays anything."
        ),
    ),
    CostFigure(
        key="part_b_immunosuppressive_drug_premium",
        label="Part B immunosuppressive drug benefit premium",
        amount=Decimal("121.60"),
        unit="per month",
        applies_to=(
            "The monthly premium of the immunosuppressive drug benefit for someone whose "
            "Medicare, held because of End-Stage Renal Disease, ended 36 months after a kidney "
            "transplant; it can be higher based on income."
        ),
        page=53,
        source_text=(
            "You’ll pay a monthly premium of $121.60 (or higher based on your income) and $283 "
            "deductible for this immunosuppressive drug benefit in 2026."
        ),
    ),
)

COST_FIGURES: dict[str, CostFigure] = {figure.key: figure for figure in _FIGURES}


def lookup_cost_figure(figure: str) -> CostFigure:
    """The figure filed under `figure`. Raises `KeyError` for a key the table does not hold."""
    return COST_FIGURES[figure]
