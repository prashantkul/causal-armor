"""Helper to run glassbox_demo scenarios and export Rich output as HTML."""

from __future__ import annotations

import asyncio
from pathlib import Path

from rich.console import Console

# Patch the demo's console with a recording one before importing
import demo.glassbox_demo as demo_mod

html_console = Console(record=True, width=100)
demo_mod.console = html_console


async def run_and_export(
    scenario: str,
    *,
    live: bool,
    output_path: Path,
) -> None:
    """Run a single scenario and write HTML."""
    html_console.clear()
    # Reset recording buffer
    html_console._record_buffer.clear()  # noqa: SLF001

    from causal_armor import CausalArmorConfig

    config = CausalArmorConfig(
        margin_tau=0.0,
        enable_sanitization=True,
        enable_cot_masking=True,
        mask_cot_for_scoring=True,
    )

    is_attack = scenario == "attack"
    live_proxy = None

    if live:
        import httpx

        from causal_armor.providers.vllm import VLLMProxyProvider

        vllm_base_url = "http://localhost:8000"
        vllm_model = "google/gemma-3-12b-it"

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{vllm_base_url}/v1/models")
            resp.raise_for_status()

        provider = VLLMProxyProvider(
            base_url=vllm_base_url, model=vllm_model
        )
        live_proxy = demo_mod.InstrumentedVLLMProxy(provider=provider)
        html_console.print(
            f"[dim]Live mode: vLLM proxy at {vllm_base_url} "
            f"({vllm_model})[/dim]"
        )

    await demo_mod.run_scenario(
        is_attack=is_attack,
        fast=not live,
        config=config,
        live_proxy=live_proxy,
    )

    if live_proxy is not None:
        await live_proxy.provider.close()

    mode = "live" if live else "mock"
    title = f"CausalArmor Demo &mdash; {mode} {scenario}"
    html = html_console.export_html(
        theme=DARK_THEME,
        inline_styles=True,
    )
    # Rich export_html returns a full HTML doc; inject our title
    html = html.replace("<head>", f"<head>\n<title>{title}</title>", 1)
    output_path.write_text(html)
    print(f"  Wrote {output_path}")


# Dark terminal theme for Rich HTML export
from rich.terminal_theme import MONOKAI as DARK_THEME  # noqa: E402


async def main() -> None:
    demo_dir = Path(__file__).parent

    pairs = [
        ("attack", False, "mock_attack.html"),
        ("benign", False, "mock_benign.html"),
        ("attack", True, "live_attack.html"),
        ("benign", True, "live_benign.html"),
    ]

    for scenario, live, filename in pairs:
        mode = "live" if live else "mock"
        print(f"Running {mode} {scenario}...")
        await run_and_export(
            scenario, live=live, output_path=demo_dir / filename
        )

    print("\nAll HTML exports complete.")


if __name__ == "__main__":
    asyncio.run(main())
