# Geoplanetical — Game Mechanics Reference

Collected 2026-11-22 from official sources (polytale.games, geoplanetical.com).
Used as domain context for training data labeling and reward shaping.

## Core Loop
1. **Budget** → set taxes/spending → apply proposal (forecast must exist first)
2. **Legislation** → file bills → vote forecast → veto/court review possible
3. **Economy** → productive investment programmes, sovereign wealth fund, banking
4. **Diplomacy / War** → army loyalty check before war goals; escalation costs
5. **Disasters & Pandemics** → configurable frequency (None/Rare/Normal/Frequent)

## Budget System
- Revenue: income tax brackets (marginal rates), VAT categories, social contributions (employee + employer, cap base, low-wage relief)
- Spending: line items in Budget → Spending tab
- Apply = single proposal combining all pending edits
- Borrowing: separate action; markets open/closed, max amount, quoted rate
- Emergency measures: money creation (raises prices), default (closes market borrowing), IMF programme
- Sovereign debt repayment: preview shows state cash after; no buffer kept

## Income Tax Editor
- Brackets: upper threshold + marginal rate per band
- Personal allowance deducted from labour income
- Thresholds follow Median wage | Prices | Frozen
- Capital income: Ordinary schedule | Separate rate | Exempt
- Test a household: enter labour + capital income → see tax by band

## Productive Investment
- Target: company, specific good, or sector
- Goals: Produce more at home | Sell more abroad | Create jobs | Reduce critical dependence
- Set max budget + duration
- Programme selects projects within constraints
- Pause / Stop new commitments / Cancel individual project (permanent)

## Banking & Finance
- Banks: capital ratio + liquidity vs minimums
- Recapitalize only (preserve ownership) vs Recapitalize and nationalize
- Lend through central bank for liquidity shortfall at solvent bank
- Sovereign wealth fund: fund value ≠ domestic currency cash
- Stock market: automatic vs manual management; sell units/shares; inject capital into listed companies

## Currency
- Can leave shared currency and create own
- Review debt implications before naming new currency

## UI Changes Since Launch (for video frame labeling)
| Date | Change |
|------|--------|
| 1.0.2 (Oct 29) | "Treasury" renamed to "State cash" |
| Oct 30 | "Resources" tab → "Staff & Budget"; GDP single figure |
| Oct 31 | "All laws" filter in Legislation; UI scale slider to 140% |
| 1.0.7 (Nov 1) | Game speeds x1/x2 slowed; x3 unchanged |

## Difficulty Settings
- Disasters: None / Rare / Normal / Frequent
- Pandemics: same options
- AI war: on/off
- AI nuclear first use: on/off
- Leader temperament: Calm / Normal / Tense (fixed at start)
- Start date: 1 Jan 2026 (no historical scenarios)

## Reward Signals for RL Agent
- Approval rating (public opinion)
- State cash balance (avoid negative)
- Debt level (lower is better unless strategic borrowing)
- Army loyalty (must stay above threshold before military actions)
- Economic indicators: GDP, employment, inflation
- Stability index
- Legislative success rate

## Key Screens (for OCR + state extraction)
1. Main dashboard (top bar: state cash, approval, stability)
2. Budget → Revenue & taxes
3. Budget → Spending
4. Budget → Summary (borrowing, emergency measures)
5. Legislation (bill list, vote forecast)
6. Business & Investment → Productive investment
7. Money & Finance → Sovereign wealth fund
8. Money & Finance → Banks and capital
9. Military / War page
10. Diplomacy map

## Sources
- https://polytale.games/geoplanetical-wiki
- https://geoplanetical.com/guides/beginner/
- https://store.steampowered.com/app/940230/Geoplanetical/
