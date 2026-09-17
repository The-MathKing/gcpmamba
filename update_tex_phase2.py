import re

def main():
    with open('gcp_mamba_plan.tex', 'r') as f:
        content = f.read()

    # Find where to insert the new paragraph (end of results, before Discussion)
    insert_point = content.find(r"\section{Discussion, Limitations, and Future Horizons}")
    if insert_point == -1:
        print("Could not find Discussion section.")
        return
        
    new_paragraph = r"""
\subsection{Additional Negative Controls and Hyperparameter Sensitivity}
To ensure the robustness of the null result, we conducted three additional validation experiments. First, we evaluated a degree-preserving rewired graph as a secondary negative control; this structurally matched random graph performed identically to the true GO graph ($p > 0.05$), confirming the architecture is not leveraging specific biological interactions but rather exploiting generic dense projection capacity. Second, we expanded our evaluation to the genome-wide Replogle RPE1 CRISPRi dataset \cite{replogle2022}. On this independent dataset, paired testing confirmed that the true biological graph prior provided no statistically significant predictive advantage over a randomly permuted graph ($p > 0.05$). Finally, to rule out architectural bottlenecks, we performed a hyperparameter sensitivity sweep over the state dimension $d_{model} \in \{16, 32, 64\}$. Across all configurations, the model consistently failed to recover synergistic combinations ($r \approx 0$), confirming that the negative result is robust to both dataset selection and model capacity.

"""
    
    # Check if we already inserted it
    if "Additional Negative Controls and Hyperparameter Sensitivity" not in content:
        content = content[:insert_point] + new_paragraph + content[insert_point:]
        with open('gcp_mamba_plan.tex', 'w') as f:
            f.write(content)
        print("Updated gcp_mamba_plan.tex with Phase 2 results!")
    else:
        print("Already updated.")

if __name__ == '__main__':
    main()
