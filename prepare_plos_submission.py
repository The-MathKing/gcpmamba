import re
import os

def convert_citations():
    with open('gcp_mamba_plan.tex', 'r') as f:
        content = f.read()
        
    # Remove the optional argument from \bibitem
    content = re.sub(r'\\bibitem\[.*?\]{(.*?)}', r'\\bibitem{\1}', content)
    
    with open('gcp_mamba_plan.tex', 'w') as f:
        f.write(content)
        
    print("Converted citations to numbered format.")

def draft_cover_letter():
    letter = """Dear Editor,

We are submitting our manuscript, "GCP-Mamba: Graph-Conditioned Perturbation Mamba," for consideration as a Research Article in PLOS Computational Biology. 

Our study rigorously investigates the integration of Gene Ontology (GO) co-expression topology into the Mamba state-space model for predicting the transcriptomic effects of genetic perturbations. We propose a plausible and well-founded architectural idea: injecting structural graph priors directly into the recurrent discretization step of the SSM.

However, after extensive empirical evaluation across multiple independent single-cell CRISPR screening datasets (Norman 2019 and Replogle 2022), and evaluating against strict permuted-graph and degree-preserving rewired controls, we present an honest, comprehensive negative result. We demonstrate that the model fails to meaningfully leverage the provided biological graph prior, predicting primarily the mean expression delta regardless of the provided perturbation. 

In light of the recent publication of Ahlmann-Eltze et al. 2025, which reported similar failures of deep learning models to outperform simple linear baselines on this task, our manuscript provides crucial complementary evidence. We deeply analyze the failure modes of graph-conditioned SSMs, demonstrating that the failure is robust to hyperparameter scaling, dataset selection, and architecture variants. We believe this work is equally or more rigorous than the original study, perfectly aligning with PLOS Computational Biology's scope for "'scooped' manuscripts that confirm, replicate, extend, or are complementary to a recently published, significant advance."

We believe our transparent reporting of this negative result provides valuable insights to the computational biology community and saves future researchers from pursuing similar flawed architectural avenues.

Thank you for your consideration.

Sincerely,
The Authors
"""
    with open('cover_letter.txt', 'w') as f:
        f.write(letter)
    print("Drafted cover_letter.txt")

def draft_author_summary():
    summary = """# Author Summary

Predicting how cells will respond to genetic modifications is a major goal in computational biology. Recently, advanced AI models have been applied to this problem, but their actual performance and ability to use biological knowledge (like gene interaction networks) remain debated. In this study, we developed a model called GCP-Mamba that directly integrates known gene networks into a modern sequence model to predict these cellular responses. We designed rigorous tests, including providing the model with fake, randomized networks, to see if it truly used the biological information. We found that the model completely failed to use the biological network, performing identically with real and fake data. It primarily predicted the average response regardless of the specific modification. Our findings provide a rigorous, transparent "negative result," complementing recent studies that question the current utility of deep learning for this task. By honestly reporting these architectural limitations, we aim to guide the community toward more effective ways of integrating biological priors into AI models.
"""
    with open('author_summary.md', 'w') as f:
        f.write(summary)
        
    print("Drafted author_summary.md")
    
    # Inject Author summary into LaTeX document
    with open('gcp_mamba_plan.tex', 'r') as f:
        content = f.read()
        
    if "Author Summary" not in content:
        insert_idx = content.find(r"\section{Introduction")
        if insert_idx != -1:
            tex_summary = r"""
\section*{Author Summary}
Predicting how cells will respond to genetic modifications is a major goal in computational biology. Recently, advanced AI models have been applied to this problem, but their actual performance and ability to use biological knowledge (like gene interaction networks) remain debated. In this study, we developed a model called GCP-Mamba that directly integrates known gene networks into a modern sequence model to predict these cellular responses. We designed rigorous tests, including providing the model with fake, randomized networks, to see if it truly used the biological information. We found that the model completely failed to use the biological network, performing identically with real and fake data. It primarily predicted the average response regardless of the specific modification. Our findings provide a rigorous, transparent ``negative result,'' complementing recent studies that question the current utility of deep learning for this task. By honestly reporting these architectural limitations, we aim to guide the community toward more effective ways of integrating biological priors into AI models.

"""
            content = content[:insert_idx] + tex_summary + content[insert_idx:]
            with open('gcp_mamba_plan.tex', 'w') as f:
                f.write(content)
            print("Injected Author Summary into LaTeX")

if __name__ == "__main__":
    convert_citations()
    draft_cover_letter()
    draft_author_summary()
