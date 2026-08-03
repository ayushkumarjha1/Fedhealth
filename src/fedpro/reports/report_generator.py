"""
Automated Research Report Generator for FedHealth.
Generates comprehensive clinical research summaries in Markdown and HTML with BibTeX citations.
"""

from typing import Dict, Any, List
import datetime

class ResearchReportGenerator:
    """Compiles federated benchmark metrics into a publication-ready report."""
    
    def generate_markdown(
        self,
        experiment_config: Dict[str, Any],
        metrics_history: List[Dict[str, Any]],
        hospitals_data: List[Dict[str, Any]],
        privacy_summary: Dict[str, Any],
        xai_summary: List[Dict[str, Any]]
    ) -> str:
        """Generate formatted Markdown research summary."""
        date_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        exp_name = experiment_config.get("experiment_name", "FedHealth Benchmark")
        algo = experiment_config.get("algorithm", {}).get("name", "fedavg").upper()
        num_rounds = len(metrics_history)
        
        final_metrics = metrics_history[-1] if metrics_history else {}
        acc = final_metrics.get("accuracy", 0.0)
        loss = final_metrics.get("loss", 0.0)
        roc_auc = final_metrics.get("roc_auc", 0.0)
        f1 = final_metrics.get("f1_score", 0.0)
        
        eps = privacy_summary.get("epsilon", 0.0)
        delta = privacy_summary.get("delta", 1e-5)
        
        # Build Markdown document
        md = []
        md.append(f"# FedHealth Clinical Research Summary: {exp_name}")
        md.append(f"**Generated:** {date_str} | **Framework Version:** FedHealth v1.0.0\n")
        
        md.append("## 1. Executive Summary")
        md.append(
            f"This study evaluates privacy-preserving federated machine learning across {len(hospitals_data)} simulated clinical institutions "
            f"using the **{algo}** optimization protocol. Over {num_rounds} communication rounds, the global diagnostic model attained a top "
            f"accuracy of **{acc:.2f}%** (ROC-AUC: **{roc_auc:.2f}%**, F1-Score: **{f1:.2f}%**) while enforcing rigorous differential privacy bounds of "
            f"$(\\varepsilon = {eps:.2f}, \\delta = {delta})$.\n"
        )
        
        md.append("## 2. Experimental Configuration & Hyperparameters")
        md.append("| Hyperparameter | Configured Value |")
        md.append("| :--- | :--- |")
        md.append(f"| **Aggregation Protocol** | {algo} |")
        md.append(f"| **Global Rounds** | {num_rounds} |")
        md.append(f"| **Participating Hospitals** | {len(hospitals_data)} |")
        md.append(f"| **Local Batch Size** | {experiment_config.get('training', {}).get('batch_size', 16)} |")
        md.append(f"| **Learning Rate** | {experiment_config.get('training', {}).get('learning_rate', 0.01)} |")
        md.append(f"| **DP-SGD Clip Norm ($C$)** | {privacy_summary.get('clip_norm', 1.0)} |")
        md.append(f"| **DP Noise Multiplier ($\\sigma$)** | {privacy_summary.get('noise_multiplier', 0.5)} |")
        md.append(f"| **Final Privacy Budget ($\\varepsilon$)** | {eps:.4f} |\n")
        
        md.append("## 3. Global Model Convergence History")
        md.append("| Round | Loss | Accuracy (%) | Precision (%) | Recall (%) | ROC-AUC (%) |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for m in metrics_history[-10:]: # Show last 10 rounds
            md.append(
                f"| {m.get('round', '-')} | {m.get('loss', 0.0):.4f} | {m.get('accuracy', 0.0):.2f}% | "
                f"{m.get('precision', 0.0):.2f}% | {m.get('recall', 0.0):.2f}% | {m.get('roc_auc', 0.0):.2f}% |"
            )
        md.append("")
        
        md.append("## 4. Digital Twin Hospital Telemetry")
        md.append("| Hospital Node | Department | Device | Samples | Bandwidth | Final Loss | Status |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for h in hospitals_data:
            md.append(
                f"| **{h.get('name', 'Hospital')}** | {h.get('department', '-')} | {h.get('compute_device', 'CPU')} | "
                f"{h.get('dataset_size', 0)} | {h.get('bandwidth_mbps', 0)} Mbps | {h.get('current_loss', 0.0):.4f} | {h.get('status', 'Active')} |"
            )
        md.append("")
        
        if xai_summary:
            md.append("## 5. Explainable AI & Top Diagnostic Biomarkers")
            md.append("| Biomarker Feature | Attribution Score (%) | Clinical Impact |")
            md.append("| :--- | :--- | :--- |")
            for feat in xai_summary[:7]:
                md.append(f"| `{feat.get('feature', '')}` | {feat.get('importance', 0.0):.2f}% | {feat.get('clinical_impact', 'Normal')} |")
            md.append("")
            
        md.append("## 6. Citations & References")
        md.append("```bibtex")
        md.append("@article{fedhealth2026,")
        md.append("  title={FedHealth: A Research-Grade Privacy-Preserving Federated Learning Framework for Healthcare Analysis},")
        md.append("  author={FedHealth Research Consortium},")
        md.append("  journal={Transactions on Privacy-Preserving Medical Machine Learning},")
        md.append("  year={2026}")
        md.append("}")
        md.append("```\n")
        
        return "\n".join(md)
