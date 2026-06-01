"""CLI entry point for g2p-rag: ingest, query, and info commands."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv
import structlog

load_dotenv()
log = structlog.get_logger()

app = typer.Typer(
    name="g2p-rag",
    help="RAG over the Broad Institute G2P portal.",
    add_completion=False,
)
console = Console()

# ---------------------------------------------------------------------------
# Reusable option types
# ---------------------------------------------------------------------------

DataDir = Annotated[
    Path,
    typer.Option("--data-dir", "-d", help="Cache and DB directory", envvar="G2P_DATA_DIR"),
]


# ---------------------------------------------------------------------------
# Version callback
# ---------------------------------------------------------------------------


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", "-V", is_eager=True),
) -> None:
    """g2p-rag: Retrieval-Augmented Generation over the Broad Institute G2P portal."""
    if version:
        from g2p_rag import __version__
        typer.echo(f"g2p-rag {__version__}")
        raise typer.Exit()


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@app.command()
def ingest(
    genes: Annotated[
        Optional[list[str]],
        typer.Option("--gene", "-g", help="Gene symbols to ingest (default: all 25)"),
    ] = None,
    data_dir: DataDir = Path("data"),
    embedding_model: Annotated[
        str,
        typer.Option("--embedding-model", "-e", help="Embedding model name"),
    ] = "sentence-transformers/all-MiniLM-L6-v2",
    force_refetch: Annotated[
        bool,
        typer.Option("--force-refetch", help="Ignore cache and re-fetch from APIs"),
    ] = False,
) -> None:
    """Fetch protein data from G2P + ClinVar, chunk, embed, and store in ChromaDB."""
    # Lazy imports — keep startup fast and give clear errors on missing deps
    try:
        from g2p_rag import fetch, chunk
    except ImportError as exc:
        console.print(f"[red]Import error:[/red] {exc}")
        console.print("Run [bold]pip install -e .[/bold] to install dependencies.")
        raise typer.Exit(1)

    try:
        from g2p_rag import retrieve
    except ImportError as exc:
        console.print(f"[red]Import error (retrieve):[/red] {exc}")
        console.print("Ensure chromadb and sentence-transformers are installed.")
        raise typer.Exit(1)

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir = data_dir / "chroma"

    # Resolve gene list
    gene_list: list[str] = genes if genes else fetch.GENE_LIST
    console.print(
        f"[bold]Ingesting[/bold] {len(gene_list)} gene(s) "
        f"→ [dim]{data_dir}[/dim]"
    )

    gene_data: dict = {}
    chunks: list = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        # Step 1 — fetch
        task = progress.add_task("Fetching data from G2P + ClinVar...", total=None)
        try:
            # Thread the --force-refetch flag through to the fetcher. This
            # used to be silently dropped (CLI flag accepted, but never
            # passed) which meant stale 404s in the cache survived across
            # ingests.
            gene_data = fetch.fetch_all_genes(
                gene_list, data_dir, force_refetch=force_refetch,
            )
        except Exception as exc:
            console.print(f"[red]Fetch failed:[/red] {exc}")
            log.exception("ingest.fetch_failed")
            raise typer.Exit(1)
        progress.update(task, description="[green]Fetch complete.[/green]")

        # Step 2 — chunk
        progress.update(task, description="Chunking proteins...")
        try:
            chunks = chunk.chunk_all(gene_data)
        except Exception as exc:
            console.print(f"[red]Chunking failed:[/red] {exc}")
            log.exception("ingest.chunk_failed")
            raise typer.Exit(1)
        progress.update(task, description="[green]Chunking complete.[/green]")

        # Step 3 — build index
        progress.update(task, description="Building index...")
        try:
            embedder = retrieve.load_embedder(embedding_model)
            retrieve.build_index(chunks, chroma_dir, embedder)
        except Exception as exc:
            console.print(f"[red]Index build failed:[/red] {exc}")
            log.exception("ingest.index_failed")
            raise typer.Exit(1)
        progress.update(task, description="[green]Index ready.[/green]")

    # Print summary table
    table = Table(title="Ingest Summary", show_lines=True)
    table.add_column("Gene", style="bold cyan")
    table.add_column("Domains", justify="right")
    table.add_column("Variants", justify="right")
    table.add_column("Total Chunks", justify="right")

    chunk_counts: dict[str, int] = {}
    for c in chunks:
        chunk_counts[c.gene] = chunk_counts.get(c.gene, 0) + 1

    for gene_symbol, data in gene_data.items():
        structure = data["structure"]
        variants = data.get("variants", [])
        n_domains = len(getattr(structure.features, "domains", []))
        n_variants = len(variants)
        n_chunks = chunk_counts.get(gene_symbol, 0)
        table.add_row(gene_symbol, str(n_domains), str(n_variants), str(n_chunks))

    console.print(table)
    console.print(
        f"\n[green]Done.[/green] {len(chunks)} chunks indexed in [dim]{chroma_dir}[/dim]"
    )


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


@app.command()
def query(
    question: Annotated[
        str,
        typer.Argument(help="Natural language question about protein structure or variants"),
    ],
    data_dir: DataDir = Path("data"),
    embedding_model: Annotated[
        str,
        typer.Option("--embedding-model", "-e"),
    ] = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-k", help="Number of chunks to retrieve"),
    ] = 5,
    generate: Annotated[
        bool,
        typer.Option("--generate/--no-generate", help="Run LLM generation (requires ANTHROPIC_API_KEY)"),
    ] = True,
    llm_model: Annotated[
        str,
        typer.Option("--llm-model", help="Anthropic model for generation"),
    ] = "claude-sonnet-4-6",
) -> None:
    """Query the G2P knowledge base with natural language."""
    try:
        from g2p_rag import retrieve
    except ImportError as exc:
        console.print(f"[red]Import error (retrieve):[/red] {exc}")
        raise typer.Exit(1)

    data_dir = Path(data_dir)
    chroma_dir = data_dir / "chroma"

    # Load retriever
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Loading retriever...", total=None)
        try:
            embedder = retrieve.load_embedder(embedding_model)
            retriever = retrieve.load_retriever(chroma_dir, embedder)
        except retrieve.CollectionEmptyError:
            console.print(
                "[yellow]No data ingested.[/yellow] "
                "Run [bold]g2p-rag ingest[/bold] first."
            )
            raise typer.Exit(1)
        except Exception as exc:
            console.print(f"[red]Failed to load retriever:[/red] {exc}")
            log.exception("query.load_retriever_failed")
            raise typer.Exit(1)

        # Retrieve
        progress.update(task, description=f"Searching top {top_k} chunks...")
        try:
            results = retriever.search(question, top_k=top_k)
        except Exception as exc:
            console.print(f"[red]Search failed:[/red] {exc}")
            log.exception("query.search_failed")
            raise typer.Exit(1)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        raise typer.Exit(0)

    # Print retrieved chunks
    console.print(f"\n[bold]Top {len(results)} retrieved chunks:[/bold]\n")
    for i, result in enumerate(results, 1):
        chunk = result.chunk
        gene = chunk.gene or "?"
        uniprot = chunk.uniprot_id or "?"
        chunk_type = chunk.chunk_type or "?"
        r_start = chunk.residue_start
        r_end = chunk.residue_end
        text = chunk.text
        excerpt = text[:300] + ("…" if len(text) > 300 else "")

        header = (
            f"[bold cyan]{gene}[/bold cyan]  "
            f"[dim]{uniprot}[/dim]  "
            f"type=[yellow]{chunk_type}[/yellow]  "
            f"residues={r_start}–{r_end}"
        )
        console.print(Panel(excerpt, title=f"[{i}] {header}", expand=False))

    # LLM generation
    if generate:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            console.print(
                "\n[yellow]Warning:[/yellow] ANTHROPIC_API_KEY not set — "
                "skipping generation. Set the key or use [bold]--no-generate[/bold]."
            )
        else:
            try:
                from g2p_rag import generate as gen_mod
            except ImportError as exc:
                console.print(f"[red]Import error (generate):[/red] {exc}")
                raise typer.Exit(1)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("Generating answer...", total=None)
                try:
                    chain = gen_mod.G2PChain(model=llm_model)
                    answer = chain.run(question=question, chunks=results)
                except Exception as exc:
                    console.print(f"[red]Generation failed:[/red] {exc}")
                    log.exception("query.generation_failed")
                    raise typer.Exit(1)

            console.print(
                Panel(answer, title="[bold green]Generated Answer[/bold green]", expand=False)
            )


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


@app.command()
def info(data_dir: DataDir = Path("data")) -> None:
    """Show information about the current index."""
    try:
        from g2p_rag import retrieve
    except ImportError as exc:
        console.print(f"[red]Import error (retrieve):[/red] {exc}")
        raise typer.Exit(1)

    data_dir = Path(data_dir)
    chroma_dir = data_dir / "chroma"

    console.print(f"[bold]Data directory:[/bold] {data_dir.resolve()}")
    console.print(f"[bold]ChromaDB path:[/bold]  {chroma_dir.resolve()}")

    if not chroma_dir.exists():
        console.print("\n[yellow]No index found.[/yellow] Run [bold]g2p-rag ingest[/bold] first.")
        return

    try:
        stats = retrieve.index_stats(chroma_dir)
    except retrieve.CollectionEmptyError:
        console.print("\n[yellow]Index exists but is empty.[/yellow] Run [bold]g2p-rag ingest[/bold].")
        return
    except Exception as exc:
        console.print(f"[red]Could not read index:[/red] {exc}")
        return

    table = Table(title="Index Information", show_lines=True)
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Collection size", str(stats.get("count", "?")))
    table.add_row("Genes indexed", ", ".join(stats.get("genes", [])) or "—")
    table.add_row("Embedding model", stats.get("embedding_model", "—"))
    table.add_row("Chunk types", ", ".join(stats.get("chunk_types", [])) or "—")
    table.add_row("Data directory", str(data_dir.resolve()))

    console.print(table)
