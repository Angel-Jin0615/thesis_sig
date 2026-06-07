# Evaluation Summary

Signature bootstrap achieves **lower linear signature MMD** than historical bootstrap in the reported test evaluation.
VaR/ES estimates are closer to the real test distribution after fixing the positive-loss convention.
Historical bootstrap remains a strong baseline.
The best overall validation config uses forward=1, which is good for daily distributional matching but may not fully preserve multi-day path dependence.
Squared-return ACF remains imperfect, so volatility clustering is only partially captured.
This is scenario generation, not directional forecasting.

## Best Overall Validation Configuration
lookback,signature_level,forward,k_neighbors,signature_mmd,volatility_error,var95_error,es95_error,drawdown_error,acf_error,normalized_signature_mmd,normalized_volatility_error,normalized_var95_error,normalized_es95_error,normalized_drawdown_error,normalized_acf_error,score,config_type
20,3,1,10,0.0005351804047955624,0.02361367766816292,0.002413381287421255,0.0008448895805903377,0.015291041649865132,0.03256461574898756,0.12062393772288203,0.13761504002340189,0.0,0.03758282073019477,0.0,0.5270327606780006,0.8228545591544794,best_overall

## Best Multiday Validation Configuration (forward >= 5)
lookback,signature_level,forward,k_neighbors,signature_mmd,volatility_error,var95_error,es95_error,drawdown_error,acf_error,normalized_signature_mmd,normalized_volatility_error,normalized_var95_error,normalized_es95_error,normalized_drawdown_error,normalized_acf_error,score,config_type
20,2,5,10,0.0005925414976210846,0.020003961151775156,0.0026521429831849125,0.0019763033435521134,0.018617325217953166,0.03827014387639471,0.1453028641681656,0.0,0.07413397943802653,0.3457200993044914,0.1953559798583378,0.9857737981097503,1.7462867208787716,best_multiday