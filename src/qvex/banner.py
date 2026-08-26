"""Q-VEX Cosmic Knowledge Graph Banner & Terminal Art."""

import os
import sys

CYAN = "\033[96m"
BLUE = "\033[94m"
PURPLE = "\033[35m"
MAGENTA = "\033[95m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

BANNER = f"""
{DIM}         .             {YELLOW}✦{RESET}{DIM}                  .             {YELLOW}*{RESET}{DIM}               {YELLOW}✦{RESET}
{DIM}    {YELLOW}*{RESET}{DIM}         {CYAN}★{RESET}{DIM}               .                 *               .{RESET}
{DIM}         ┌──────────{CYAN}★ (BM25 Seed){RESET}{DIM} ───────────┐                 {YELLOW}✦{RESET}
{DIM}         │         / \\                           │          .               *{RESET}
{DIM}         │        /   \\  {MAGENTA}[CTE Graph Walk]{RESET}{DIM}        │                    .{RESET}
{DIM}         ▼       /     ▼                         ▼             {GREEN}★ (Context){RESET}
{DIM}    {BLUE}★ (FTS5){RESET}{DIM} ──┼───────{CYAN}★ (Entity Overlap){RESET}{DIM} ──────┼──────{PURPLE}★ (4-bit TurboQuant){RESET}
{DIM}         │     /         \\                       │     /
{DIM}         │    /           \\                      │    /   {MAGENTA}[k-Hop Expansion]{RESET}
{DIM}         ▼   ▼             ▼                     ▼   ▼
{DIM}       {GREEN}★ (Agent Memory){RESET}{DIM} ──── {CYAN}★ (Shared State){RESET}{DIM} ────{PURPLE}★ (Vector Rerank){RESET}
{DIM}    .             *                {YELLOW}✦{RESET}{DIM}              .              ✦{RESET}

{CYAN}{BOLD}     ██████╗        {MAGENTA} ██╗   ██╗███████╗██╗  ██╗{RESET}
{CYAN}{BOLD}    ██╔═══██╗       {MAGENTA} ██║   ██║██╔════╝╚██╗██╔╝{RESET}
{CYAN}{BOLD}    ██║   ██║ █████╗{MAGENTA} ██║   ██║█████╗   ╚███╔╝ {RESET}
{CYAN}{BOLD}    ██║▄▄ ██║ ╚════╝{MAGENTA} ╚██╗ ██╔╝██╔══╝   ██╔██╗ {RESET}
{CYAN}{BOLD}    ╚██████╔╝       {MAGENTA}  ╚████╔╝ ███████╗██╔╝ ██╗{RESET}
{CYAN}{BOLD}     ╚══▀▀═╝        {MAGENTA}   ╚═══╝  ╚══════╝╚═╝  ╚═╝{RESET}

{BOLD}    ✦ Hyper-Compressed Graph-Vector Database Engine v0.3.0 ✦{RESET}
{DIM}    [SQLite FTS5] • [Recursive CTEs] • [2/4-bit TurboQuant] • [Multi-Agent Swarms]{RESET}
"""

def show_banner() -> None:
    """Print the Q-VEX cosmic knowledge graph constellation banner safely across platforms."""
    try:
        if os.name == "nt":
            os.system("")  # Enable ANSI color escape codes on Windows
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    
    try:
        print(BANNER)
    except Exception:
        # Robust fallback for environments without full unicode encoding
        ascii_fallback = (
            BANNER.replace("✦", "*")
            .replace("★", "*")
            .replace("█", "#")
            .replace("╔", "+")
            .replace("╗", "+")
            .replace("╚", "+")
            .replace("╝", "+")
            .replace("═", "-")
            .replace("║", "|")
            .replace("┌", "+")
            .replace("┐", "+")
            .replace("│", "|")
            .replace("─", "-")
            .replace("┼", "+")
        )
        print(ascii_fallback)
