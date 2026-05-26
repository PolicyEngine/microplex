# PolicyEngine-US-Data Comparison

_Generated: 2025-12-27 22:03_

## Dataset Summary

| Dataset | Records | Weighted Population |
|---------|---------|---------------------|
| PE Enhanced CPS | 144,265 | 324,365,066 |
| PolicyEngine CPS | 142,125 | 337,689,642 |

## Income Aggregates Comparison

| Variable | PE Enhanced CPS | PolicyEngine CPS | Ratio | Status |
|----------|-----------------|--------------|-------|--------|
| Employment Income | 11,595B | 12,072B | 1.04 | ✅ |
| Self-Employment | 617B | 532B | 0.86 | ⚠️ |
| Interest Income | 824B | 817B | 0.99 | ✅ |
| Dividend Income | 238B | 218B | 0.92 | ✅ |
| Capital Gains | 273B | - | - | ⏳ |
| Social Security | 1,171B | 1,226B | 1.05 | ✅ |
| Rental Income | 268B | 239B | 0.89 | ⚠️ |
| SSI | 58B | 59B | 1.00 | ✅ |
| Unemployment Comp | 22B | 25B | 1.14 | ⚠️ |

## Demographics Comparison

| Age Group | PE Enhanced CPS | PolicyEngine CPS | Ratio |
|-----------|-----------------|--------------|-------|
| Under 18 | 69.2M | 73.0M | 1.05 |
| 18-64 | 194.9M | 203.2M | 1.04 |
| 65+ | 60.2M | 61.5M | 1.02 |

## Key Findings

### Strong Alignment (ratio 0.9-1.1)

- **Employment Income**: 1.04 ratio - Core wage/salary income matches well
- **Interest Income**: 0.99 ratio - Near perfect when comparing totals
- **Dividend Income**: 0.92 ratio - Close match for total dividends
- **Social Security**: 1.05 ratio - Benefits align well
- **SSI**: 1.00 ratio - Perfect match

### Moderate Differences (ratio 0.7-0.9 or 1.1-1.3)

- **Self-Employment**: 0.86 ratio - PE has higher totals (~15% more)
- **Rental Income**: 0.89 ratio - PE slightly higher
- **Unemployment**: 1.14 ratio - PolicyEngine slightly higher

### Missing in PolicyEngine CPS

- Capital gains (PE has 273B total)
- Separate taxable/tax-exempt breakdowns
- Qualified vs non-qualified dividend split

### Weight Differences

PE Enhanced CPS weights to 324M population while PolicyEngine CPS weights to 338M.
This 4% difference affects all aggregate comparisons.
