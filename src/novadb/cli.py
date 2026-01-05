"""Command-line interface for NovaDB.

Provides CLI commands for:
- Processing structures
- Running genetic search
- Generating datasets
- Managing storage
- S3 streaming and sync
- Distributed processing
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional

from novadb.config import Config
from novadb.pipeline.pipeline import DataPipeline, create_pipeline


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def cmd_process(args: argparse.Namespace) -> int:
    """Process structures command."""
    setup_logging(args.verbose)
    logger = logging.getLogger("novadb.cli")

    logger.info(f"Processing structures from {args.input}")

    # Create pipeline
    pipeline = create_pipeline(args.config)

    # Set up MSA search if enabled
    if args.msa:
        pipeline.setup_msa_search(
            jackhmmer_binary=args.jackhmmer or "jackhmmer",
            hhblits_binary=args.hhblits or "hhblits",
        )

    # Set up template search if enabled
    if args.templates:
        pipeline.setup_template_search(
            pdb_database=args.pdb_db or "",
            pdb_mmcif_dir=args.pdb_mmcif or "",
        )

    # Process
    if Path(args.input).is_file():
        # Single file
        sample = pipeline.process_structure(
            args.input,
            run_msa=args.msa,
            run_templates=args.templates,
        )
        if sample:
            logger.info(f"Processed {sample.pdb_id}: {sample.features.num_tokens} tokens")
        else:
            logger.error("Processing failed")
            return 1
    else:
        # Directory
        stats = pipeline.process_directory(
            args.input,
            output_prefix=args.output,
            max_structures=args.max,
        )
        logger.info(f"Processed {stats.processed_successfully}/{stats.total_structures} structures")
        if stats.failed > 0:
            logger.warning(f"Failed: {stats.failed}")

    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Run genetic search command."""
    setup_logging(args.verbose)
    logger = logging.getLogger("novadb.cli")

    from novadb.search.msa.jackhmmer import JackhmmerSearch
    from novadb.search.msa.hhblits import HHBlitsSearch

    # Read sequence
    sequence = args.sequence
    if Path(args.sequence).exists():
        with open(args.sequence) as f:
            lines = f.readlines()
            sequence = "".join(
                line.strip() for line in lines if not line.startswith(">")
            )

    logger.info(f"Searching with sequence of length {len(sequence)}")

    # Run search based on tool
    if args.tool == "jackhmmer":
        searcher = JackhmmerSearch(binary_path=args.binary or "jackhmmer")
        msa = searcher.search(sequence, args.database)
    elif args.tool == "hhblits":
        searcher = HHBlitsSearch(binary_path=args.binary or "hhblits")
        msa = searcher.search(sequence, args.database)
    else:
        logger.error(f"Unknown tool: {args.tool}")
        return 1

    logger.info(f"Found {len(msa.sequences)} sequences")

    # Save output
    output_path = Path(args.output)
    if args.format == "a3m":
        output_path.write_text(msa.to_a3m())
    else:
        output_path.write_text(msa.to_stockholm())

    logger.info(f"Saved to {args.output}")
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    """Generate dataset command."""
    setup_logging(args.verbose)
    logger = logging.getLogger("novadb.cli")

    from novadb.processing.curation.sampling import DatasetSampler, SamplingConfig
    from novadb.storage.backends import LocalStorage
    from novadb.storage.serialization import DatasetReader

    if args.subcmd == "stats":
        # Show dataset statistics
        storage = LocalStorage(args.path)
        reader = DatasetReader(storage)
        logger.info(f"Dataset: {args.path}")
        logger.info(f"  Samples: {reader.num_samples}")
        logger.info(f"  Shards: {reader.num_shards}")

    elif args.subcmd == "sample":
        # Sample from dataset
        sampler = DatasetSampler(seed=args.seed)

        # Load entries from index
        # (Would need to implement this based on actual dataset format)
        logger.info(f"Sampling {args.count} entries")

    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Configuration command."""
    setup_logging(args.verbose)

    if args.subcmd == "show":
        config = Config()
        import yaml
        print(yaml.dump(config.to_dict(), default_flow_style=False))

    elif args.subcmd == "init":
        config = Config()
        output_path = Path(args.output or "novadb.yaml")
        config.to_yaml(str(output_path))
        print(f"Created configuration file: {output_path}")

    return 0


def cmd_s3(args: argparse.Namespace) -> int:
    """S3 operations command."""
    setup_logging(args.verbose)
    logger = logging.getLogger("novadb.cli")

    from novadb.storage.streaming import (
        S3StreamConfig,
        S3StreamingProcessor,
        S3SyncManager,
        RetryConfig,
    )

    if args.subcmd == "process":
        # Process structures from S3
        config = S3StreamConfig(
            aws_region=args.region or "us-east-1",
            max_concurrent=args.workers,
            batch_size=args.batch_size,
            retry=RetryConfig(max_retries=args.retries),
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            enable_dlq=args.dlq,
        )

        # Parse S3 URIs
        source_parts = args.input.replace("s3://", "").split("/", 1)
        source_bucket = source_parts[0]
        source_prefix = source_parts[1] if len(source_parts) > 1 else ""

        output_parts = args.output.replace("s3://", "").split("/", 1)
        output_bucket = output_parts[0]
        output_prefix = output_parts[1] if len(output_parts) > 1 else ""

        # Import processing function
        from novadb.data.parsers.mmcif_parser import MMCIFParser
        from novadb.processing.features.features import FeatureExtractor

        parser = MMCIFParser()
        extractor = FeatureExtractor()

        def process_structure(key: str, data: bytes) -> dict:
            """Process a single structure."""
            structure = parser.parse_bytes(data, key)
            if structure is None:
                return {"error": "Failed to parse structure"}
            features = extractor.extract(structure)
            return features.to_dict()

        async def run():
            processor = S3StreamingProcessor(config)
            resume_path = Path(args.resume) if args.resume else None

            stats = await processor.process_bucket(
                source_bucket=source_bucket,
                source_prefix=source_prefix,
                output_bucket=output_bucket,
                output_prefix=output_prefix,
                process_func=process_structure,
                resume_from=resume_path,
            )

            logger.info(f"Processing complete:")
            logger.info(f"  Total: {stats.total_items}")
            logger.info(f"  Processed: {stats.processed}")
            logger.info(f"  Failed: {stats.failed}")
            logger.info(f"  Rate: {stats.items_per_second:.1f}/s")
            logger.info(f"  Downloaded: {stats.bytes_downloaded / 1024 / 1024:.1f} MB")
            logger.info(f"  Uploaded: {stats.bytes_uploaded / 1024 / 1024:.1f} MB")

        asyncio.run(run())

    elif args.subcmd == "sync":
        # Sync between S3 and local
        config = S3StreamConfig(
            aws_region=args.region or "us-east-1",
            max_concurrent=args.workers,
        )
        sync_manager = S3SyncManager(config)

        async def run():
            if args.direction == "download":
                # S3 to local
                parts = args.source.replace("s3://", "").split("/", 1)
                bucket = parts[0]
                prefix = parts[1] if len(parts) > 1 else ""

                stats = await sync_manager.sync_to_local(
                    bucket=bucket,
                    prefix=prefix,
                    local_path=Path(args.dest),
                    overwrite=args.overwrite,
                )
            else:
                # Local to S3
                parts = args.dest.replace("s3://", "").split("/", 1)
                bucket = parts[0]
                prefix = parts[1] if len(parts) > 1 else ""

                stats = await sync_manager.sync_to_s3(
                    local_path=Path(args.source),
                    bucket=bucket,
                    prefix=prefix,
                    overwrite=args.overwrite,
                )

            logger.info(f"Sync complete:")
            logger.info(f"  Total: {stats.total_items}")
            logger.info(f"  Synced: {stats.processed}")
            logger.info(f"  Failed: {stats.failed}")

        asyncio.run(run())

    elif args.subcmd == "list":
        # List S3 objects
        config = S3StreamConfig(aws_region=args.region or "us-east-1")

        parts = args.path.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        async def run():
            processor = S3StreamingProcessor(config)
            async with processor:
                count = 0
                async for obj in processor.list_objects(bucket, prefix):
                    print(f"{obj['Key']} ({obj['Size']} bytes)")
                    count += 1
                    if args.max and count >= args.max:
                        break
                print(f"\nTotal: {count} objects")

        asyncio.run(run())

    return 0


def cmd_distributed(args: argparse.Namespace) -> int:
    """Distributed processing command."""
    setup_logging(args.verbose)
    logger = logging.getLogger("novadb.cli")

    from novadb.storage.distributed import (
        DistributedConfig,
        DistributedCoordinator,
        DistributedWorker,
        ProgressAggregator,
        Task,
    )

    config = DistributedConfig(
        redis_url=args.redis,
        max_tasks_per_worker=args.max_tasks,
    )

    if args.subcmd == "submit":
        # Submit a new job
        source_parts = args.input.replace("s3://", "").split("/", 1)
        source_bucket = source_parts[0]
        source_prefix = source_parts[1] if len(source_parts) > 1 else ""

        output_parts = args.output.replace("s3://", "").split("/", 1)
        output_bucket = output_parts[0]
        output_prefix = output_parts[1] if len(output_parts) > 1 else ""

        async def run():
            coordinator = DistributedCoordinator(config)

            job_id = await coordinator.submit_job(
                name=args.name or "novadb-processing",
                source_bucket=source_bucket,
                source_prefix=source_prefix,
                output_bucket=output_bucket,
                output_prefix=output_prefix,
            )

            print(f"Submitted job: {job_id}")
            print(f"  Source: s3://{source_bucket}/{source_prefix}")
            print(f"  Output: s3://{output_bucket}/{output_prefix}")
            print(f"\nStart workers with:")
            print(f"  novadb distributed worker --redis {args.redis} --job {job_id}")

            await coordinator.close()

        asyncio.run(run())

    elif args.subcmd == "worker":
        # Start a worker
        from novadb.data.parsers.mmcif_parser import MMCIFParser
        from novadb.processing.features.features import FeatureExtractor
        from novadb.storage.streaming import S3StreamingProcessor, S3StreamConfig

        parser = MMCIFParser()
        extractor = FeatureExtractor()
        s3_config = S3StreamConfig()

        async def process_task(task: Task) -> dict:
            """Process a distributed task."""
            payload = task.payload

            if payload.get("type") == "scan":
                # Scan task - list objects and create processing tasks
                return {"status": "scan_complete"}

            elif payload.get("type") == "process":
                # Process a single structure
                processor = S3StreamingProcessor(s3_config)
                async with processor:
                    data = await processor.download_object(
                        payload["source_bucket"],
                        payload["source_key"],
                    )

                structure = parser.parse_bytes(data, payload["source_key"])
                if structure is None:
                    return {"error": "Failed to parse structure"}

                features = extractor.extract(structure)
                return features.to_dict()

            return {"error": f"Unknown task type: {payload.get('type')}"}

        async def run():
            worker = DistributedWorker(config, process_task)
            job_ids = [args.job] if args.job else None
            await worker.run(job_ids=job_ids)

        asyncio.run(run())

    elif args.subcmd == "status":
        # Show job status
        async def run():
            coordinator = DistributedCoordinator(config)
            aggregator = ProgressAggregator(coordinator)

            if args.job:
                # Show specific job
                progress = await aggregator.get_job_progress(args.job)
                if progress:
                    print(f"Job: {progress['job_id']} ({progress['name']})")
                    print(f"  Status: {progress['status']}")
                    print(f"  Progress: {progress['completed_tasks']}/{progress['total_tasks']} "
                          f"({progress['progress_percent']:.1f}%)")
                    print(f"  Failed: {progress['failed_tasks']}")
                    if progress['eta_seconds']:
                        print(f"  ETA: {progress['eta_seconds']:.0f}s")
                else:
                    print(f"Job not found: {args.job}")
            else:
                # Show overall status
                overall = await aggregator.get_overall_progress()
                print(f"Overall Progress:")
                print(f"  Jobs: {overall['running_jobs']} running, "
                      f"{overall['completed_jobs']} completed, "
                      f"{overall['total_jobs']} total")
                print(f"  Tasks: {overall['completed_tasks']}/{overall['total_tasks']} "
                      f"({overall['progress_percent']:.1f}%)")
                print(f"  Workers: {overall['active_workers']} active, "
                      f"{overall['total_workers']} total")

            await coordinator.close()

        asyncio.run(run())

    elif args.subcmd == "jobs":
        # List jobs
        async def run():
            coordinator = DistributedCoordinator(config)
            jobs = await coordinator.list_jobs(status=args.status, limit=args.limit)

            print(f"{'Job ID':<12} {'Name':<20} {'Status':<15} {'Progress':<15}")
            print("-" * 65)
            for job in jobs:
                progress = f"{job.completed_tasks}/{job.total_tasks}"
                print(f"{job.job_id:<12} {job.name:<20} {job.status:<15} {progress:<15}")

            await coordinator.close()

        asyncio.run(run())

    elif args.subcmd == "workers":
        # List workers
        async def run():
            coordinator = DistributedCoordinator(config)
            workers = await coordinator.get_workers()

            print(f"{'Worker ID':<30} {'Status':<10} {'Tasks':<10} {'Completed':<10}")
            print("-" * 65)
            for worker in workers:
                print(f"{worker.worker_id:<30} {worker.status.value:<10} "
                      f"{len(worker.current_tasks):<10} {worker.tasks_completed:<10}")

            await coordinator.close()

        asyncio.run(run())

    elif args.subcmd == "cancel":
        # Cancel a job
        async def run():
            coordinator = DistributedCoordinator(config)
            success = await coordinator.cancel_job(args.job)
            if success:
                print(f"Cancelled job: {args.job}")
            else:
                print(f"Job not found: {args.job}")
            await coordinator.close()

        asyncio.run(run())

    elif args.subcmd == "cleanup":
        # Clean up dead workers
        async def run():
            coordinator = DistributedCoordinator(config)
            cleaned = await coordinator.cleanup_dead_workers()
            print(f"Cleaned up {cleaned} dead workers")
            await coordinator.close()

        asyncio.run(run())

    return 0


def cmd_download(args: argparse.Namespace) -> int:
    """Download data from public databases command."""
    setup_logging(args.verbose)
    logger = logging.getLogger("novadb.cli")

    from novadb.data.fetchers import (
        StorageTarget,
        FetcherConfig,
        PDBFetcher,
        PDBSeqresFetcher,
        ObsoletePDBFetcher,
        PDBConfig,
        UniProtFetcher,
        UniRefFetcher,
        BFDFetcher,
        MGnifyFetcher,
        UniProtConfig,
        RNACentralFetcher,
        RfamFetcher,
        NCBINucleotideFetcher,
        RNAConfig,
        AlphaFoldDBFetcher,
        AlphaFoldLatestFetcher,
        AlphaFoldConfig,
        CCDFetcher,
        PRDFetcher,
        AminoAcidVariantsFetcher,
        CCDConfig,
    )

    # Determine storage target
    output = getattr(args, "output", None)
    s3_bucket = getattr(args, "s3_bucket", None)
    
    if output and output.startswith("s3://"):
        storage_target = StorageTarget.S3
        parts = output.replace("s3://", "").split("/", 1)
        s3_bucket = parts[0]
        s3_prefix = parts[1] if len(parts) > 1 else ""
        local_path = None
    elif s3_bucket:
        storage_target = StorageTarget.S3
        s3_prefix = getattr(args, "s3_prefix", "")
        local_path = None
    else:
        storage_target = StorageTarget.LOCAL
        local_path = Path(output or "./data")
        s3_bucket = None
        s3_prefix = ""

    # Base config
    base_config = FetcherConfig(
        storage_target=storage_target,
        local_path=local_path,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        s3_region=getattr(args, "s3_region", "us-east-1"),
        max_concurrent=getattr(args, "workers", 10),
        max_retries=getattr(args, "retries", 3),
        verify_checksum=not getattr(args, "no_verify", False),
    )

    async def run_download():
        if args.subcmd == "pdb":
            config = PDBConfig(
                **{k: v for k, v in base_config.__dict__.items()},
                format=getattr(args, "format", "mmcif"),
            )

            if getattr(args, "seqres", False):
                logger.info("Downloading PDB SEQRES database...")
                async with PDBSeqresFetcher(config) as fetcher:
                    result = await fetcher.fetch_seqres()
                    if result.success:
                        logger.info(f"Downloaded to: {result.path}")
                    else:
                        logger.error(f"Failed: {result.error}")

            elif getattr(args, "obsolete", False):
                logger.info("Downloading obsolete PDB list...")
                async with ObsoletePDBFetcher(config) as fetcher:
                    result = await fetcher.fetch_obsolete()
                    if result.success:
                        logger.info(f"Downloaded to: {result.path}")
                    else:
                        logger.error(f"Failed: {result.error}")

            elif getattr(args, "ids", None):
                logger.info(f"Downloading {len(args.ids)} PDB entries...")
                async with PDBFetcher(config) as fetcher:
                    results = await fetcher.fetch_entries(args.ids)
                    successful = sum(1 for r in results if r.success)
                    logger.info(f"Downloaded {successful}/{len(results)} entries")

            elif getattr(args, "since", None):
                logger.info(f"Downloading PDB entries since {args.since}...")
                async with PDBFetcher(config) as fetcher:
                    entries = await fetcher.fetch_since(args.since)
                    logger.info(f"Downloaded {len(entries)} entries")

            elif getattr(args, "all", False):
                logger.info("Downloading entire PDB (this may take a long time)...")
                async with PDBFetcher(config) as fetcher:
                    stats = await fetcher.fetch_all()
                    logger.info(f"Downloaded {stats.total_downloaded} entries")
                    logger.info(f"Total size: {stats.total_size / 1024 / 1024 / 1024:.2f} GB")
            else:
                logger.error("Please specify --ids, --since, --all, --seqres, or --obsolete")
                return 1

        elif args.subcmd == "uniprot":
            database = getattr(args, "database", "swissprot")
            
            if database in ("swissprot", "trembl"):
                config = UniProtConfig(
                    **{k: v for k, v in base_config.__dict__.items()},
                    database=database,
                )
                logger.info(f"Downloading UniProt {database}...")
                async with UniProtFetcher(config) as fetcher:
                    result = await fetcher.fetch_database()
                    if result.success:
                        logger.info(f"Downloaded to: {result.path}")
                    else:
                        logger.error(f"Failed: {result.error}")

            elif database.startswith("uniref"):
                cluster_level = int(database.replace("uniref", ""))
                config = UniProtConfig(
                    **{k: v for k, v in base_config.__dict__.items()},
                )
                logger.info(f"Downloading UniRef{cluster_level}...")
                async with UniRefFetcher(config) as fetcher:
                    result = await fetcher.fetch_database(cluster_level)
                    if result.success:
                        logger.info(f"Downloaded to: {result.path}")
                    else:
                        logger.error(f"Failed: {result.error}")

            elif database == "bfd":
                logger.info("Downloading BFD (~1.7TB, this will take a long time)...")
                async with BFDFetcher(base_config) as fetcher:
                    stats = await fetcher.fetch_database()
                    logger.info(f"Downloaded {stats.total_downloaded} files")
                    logger.info(f"Total size: {stats.total_size / 1024 / 1024 / 1024:.2f} GB")

            elif database == "mgnify":
                version = getattr(args, "mgnify_version", "2022_05")
                logger.info(f"Downloading MGnify {version}...")
                async with MGnifyFetcher(base_config) as fetcher:
                    result = await fetcher.fetch_clusters(version)
                    if result.success:
                        logger.info(f"Downloaded to: {result.path}")
                    else:
                        logger.error(f"Failed: {result.error}")

        elif args.subcmd == "rna":
            database = getattr(args, "database", "rnacentral")
            
            if database == "rnacentral":
                config = RNAConfig(**{k: v for k, v in base_config.__dict__.items()})
                logger.info("Downloading RNACentral...")
                async with RNACentralFetcher(config) as fetcher:
                    result = await fetcher.fetch_database()
                    if result.success:
                        logger.info(f"Downloaded to: {result.path}")
                    else:
                        logger.error(f"Failed: {result.error}")

            elif database == "rfam":
                config = RNAConfig(
                    **{k: v for k, v in base_config.__dict__.items()},
                    download_cm=getattr(args, "rfam_cm", True),
                )
                logger.info("Downloading Rfam...")
                async with RfamFetcher(config) as fetcher:
                    if config.download_cm:
                        result = await fetcher.fetch_cm()
                        if result.success:
                            logger.info(f"CM downloaded to: {result.path}")
                        else:
                            logger.error(f"CM failed: {result.error}")
                    
                    result = await fetcher.fetch_seed()
                    if result.success:
                        logger.info(f"Seed downloaded to: {result.path}")
                    else:
                        logger.error(f"Seed failed: {result.error}")

            elif database == "nt":
                logger.info("Downloading NCBI nt (~300GB, this will take a long time)...")
                async with NCBINucleotideFetcher(base_config) as fetcher:
                    stats = await fetcher.fetch_database()
                    logger.info(f"Downloaded {stats.total_downloaded} files")
                    logger.info(f"Total size: {stats.total_size / 1024 / 1024 / 1024:.2f} GB")

        elif args.subcmd == "alphafold":
            config = AlphaFoldConfig(
                **{k: v for k, v in base_config.__dict__.items()},
                version=getattr(args, "version", "v4"),
                include_pae=getattr(args, "include_pae", False),
            )

            if getattr(args, "swissprot", False):
                logger.info("Downloading AlphaFold Swiss-Prot predictions...")
                async with AlphaFoldLatestFetcher(config) as fetcher:
                    result = await fetcher.fetch_swissprot()
                    if result.success:
                        logger.info(f"Downloaded to: {result.path}")
                    else:
                        logger.error(f"Failed: {result.error}")

            elif getattr(args, "proteome", None):
                logger.info(f"Downloading proteomes: {', '.join(args.proteome)}")
                async with AlphaFoldDBFetcher(config) as fetcher:
                    for proteome in args.proteome:
                        logger.info(f"Downloading {proteome} proteome...")
                        stats = await fetcher.fetch_proteome(
                            proteome,
                            include_pae=config.include_pae,
                        )
                        logger.info(f"  Downloaded {stats.total_downloaded} files")

            elif getattr(args, "ids", None):
                logger.info(f"Downloading {len(args.ids)} AlphaFold predictions...")
                async with AlphaFoldDBFetcher(config) as fetcher:
                    for uniprot_id in args.ids:
                        results = await fetcher.fetch_prediction(
                            uniprot_id,
                            include_pae=config.include_pae,
                        )
                        if results["structure"].success:
                            logger.info(f"  {uniprot_id}: OK")
                        else:
                            logger.error(f"  {uniprot_id}: {results['structure'].error}")
            else:
                logger.error("Please specify --ids, --proteome, or --swissprot")
                return 1

        elif args.subcmd == "ccd":
            config = CCDConfig(**{k: v for k, v in base_config.__dict__.items()})

            if getattr(args, "all", False):
                logger.info("Downloading CCD components.cif...")
                async with CCDFetcher(config) as fetcher:
                    result = await fetcher.fetch_components()
                    if result.success:
                        logger.info(f"Downloaded to: {result.path}")
                    else:
                        logger.error(f"Failed: {result.error}")

            if getattr(args, "prd", False):
                logger.info("Downloading PRD...")
                async with PRDFetcher(config) as fetcher:
                    stats = await fetcher.fetch_all()
                    logger.info(f"Downloaded {stats.total_downloaded} files")

            if getattr(args, "aa_variants", False):
                logger.info("Downloading amino acid variants...")
                async with AminoAcidVariantsFetcher(config) as fetcher:
                    result = await fetcher.fetch_variants()
                    if result.success:
                        logger.info(f"Downloaded to: {result.path}")
                    else:
                        logger.error(f"Failed: {result.error}")

            if getattr(args, "components", None):
                logger.info(f"Downloading {len(args.components)} components...")
                async with CCDFetcher(config) as fetcher:
                    results = await fetcher.fetch_components_batch(args.components)
                    successful = sum(1 for r in results if r.success)
                    logger.info(f"Downloaded {successful}/{len(results)} components")

            if not any([
                getattr(args, "all", False),
                getattr(args, "prd", False),
                getattr(args, "aa_variants", False),
                getattr(args, "components", None),
            ]):
                logger.error("Please specify --all, --prd, --aa-variants, or --components")
                return 1

        else:
            logger.error(f"Unknown subcmd: {args.subcmd}")
            return 1

        return 0

    return asyncio.run(run_download())


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="novadb",
        description="NovaDB - Biomolecular Structure Data Processing",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to configuration file",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Process command
    proc_parser = subparsers.add_parser(
        "process",
        help="Process structures",
    )
    proc_parser.add_argument(
        "input",
        help="Input mmCIF file or directory",
    )
    proc_parser.add_argument(
        "-o", "--output",
        default="processed",
        help="Output prefix",
    )
    proc_parser.add_argument(
        "--max",
        type=int,
        help="Maximum structures to process",
    )
    proc_parser.add_argument(
        "--msa",
        action="store_true",
        help="Run MSA search",
    )
    proc_parser.add_argument(
        "--templates",
        action="store_true",
        help="Run template search",
    )
    proc_parser.add_argument(
        "--jackhmmer",
        help="Path to jackhmmer binary",
    )
    proc_parser.add_argument(
        "--hhblits",
        help="Path to hhblits binary",
    )
    proc_parser.add_argument(
        "--pdb-db",
        help="Path to PDB sequence database",
    )
    proc_parser.add_argument(
        "--pdb-mmcif",
        help="Path to PDB mmCIF directory",
    )
    proc_parser.set_defaults(func=cmd_process)

    # Search command
    search_parser = subparsers.add_parser(
        "search",
        help="Run genetic search",
    )
    search_parser.add_argument(
        "sequence",
        help="Sequence or FASTA file",
    )
    search_parser.add_argument(
        "-d", "--database",
        required=True,
        help="Database path",
    )
    search_parser.add_argument(
        "-t", "--tool",
        choices=["jackhmmer", "hhblits"],
        default="jackhmmer",
        help="Search tool",
    )
    search_parser.add_argument(
        "-b", "--binary",
        help="Path to binary",
    )
    search_parser.add_argument(
        "-o", "--output",
        default="output.a3m",
        help="Output file",
    )
    search_parser.add_argument(
        "-f", "--format",
        choices=["a3m", "stockholm"],
        default="a3m",
        help="Output format",
    )
    search_parser.set_defaults(func=cmd_search)

    # Dataset command
    ds_parser = subparsers.add_parser(
        "dataset",
        help="Dataset operations",
    )
    ds_subparsers = ds_parser.add_subparsers(dest="subcmd")

    ds_stats = ds_subparsers.add_parser("stats", help="Show dataset statistics")
    ds_stats.add_argument("path", help="Dataset path")

    ds_sample = ds_subparsers.add_parser("sample", help="Sample from dataset")
    ds_sample.add_argument("path", help="Dataset path")
    ds_sample.add_argument("-n", "--count", type=int, default=10)
    ds_sample.add_argument("--seed", type=int, default=42)

    ds_parser.set_defaults(func=cmd_dataset)

    # Config command
    cfg_parser = subparsers.add_parser(
        "config",
        help="Configuration operations",
    )
    cfg_subparsers = cfg_parser.add_subparsers(dest="subcmd")

    cfg_show = cfg_subparsers.add_parser("show", help="Show configuration")
    cfg_init = cfg_subparsers.add_parser("init", help="Initialize config file")
    cfg_init.add_argument("-o", "--output", help="Output file path")

    cfg_parser.set_defaults(func=cmd_config)

    # S3 command
    s3_parser = subparsers.add_parser(
        "s3",
        help="S3 streaming operations",
    )
    s3_subparsers = s3_parser.add_subparsers(dest="subcmd")

    # S3 process
    s3_process = s3_subparsers.add_parser(
        "process",
        help="Process structures from S3",
    )
    s3_process.add_argument(
        "input",
        help="Source S3 URI (s3://bucket/prefix)",
    )
    s3_process.add_argument(
        "-o", "--output",
        required=True,
        help="Output S3 URI (s3://bucket/prefix)",
    )
    s3_process.add_argument(
        "-w", "--workers",
        type=int,
        default=50,
        help="Number of concurrent workers",
    )
    s3_process.add_argument(
        "-b", "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing",
    )
    s3_process.add_argument(
        "-r", "--retries",
        type=int,
        default=3,
        help="Maximum retry attempts",
    )
    s3_process.add_argument(
        "--checkpoint",
        help="Path to save checkpoints",
    )
    s3_process.add_argument(
        "--resume",
        help="Path to checkpoint file to resume from",
    )
    s3_process.add_argument(
        "--region",
        help="AWS region",
    )
    s3_process.add_argument(
        "--dlq",
        action="store_true",
        help="Enable dead letter queue",
    )

    # S3 sync
    s3_sync = s3_subparsers.add_parser(
        "sync",
        help="Sync between S3 and local storage",
    )
    s3_sync.add_argument(
        "source",
        help="Source path (S3 URI or local path)",
    )
    s3_sync.add_argument(
        "dest",
        help="Destination path (S3 URI or local path)",
    )
    s3_sync.add_argument(
        "-d", "--direction",
        choices=["download", "upload"],
        default="download",
        help="Sync direction",
    )
    s3_sync.add_argument(
        "-w", "--workers",
        type=int,
        default=20,
        help="Number of concurrent workers",
    )
    s3_sync.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    s3_sync.add_argument(
        "--region",
        help="AWS region",
    )

    # S3 list
    s3_list = s3_subparsers.add_parser(
        "list",
        help="List S3 objects",
    )
    s3_list.add_argument(
        "path",
        help="S3 URI to list (s3://bucket/prefix)",
    )
    s3_list.add_argument(
        "-m", "--max",
        type=int,
        help="Maximum objects to list",
    )
    s3_list.add_argument(
        "--region",
        help="AWS region",
    )

    s3_parser.set_defaults(func=cmd_s3)

    # Distributed command
    dist_parser = subparsers.add_parser(
        "distributed",
        help="Distributed processing operations",
    )
    dist_parser.add_argument(
        "--redis",
        default="redis://localhost:6379",
        help="Redis URL for coordination",
    )
    dist_parser.add_argument(
        "--max-tasks",
        type=int,
        default=10,
        help="Maximum tasks per worker",
    )
    dist_subparsers = dist_parser.add_subparsers(dest="subcmd")

    # Distributed submit
    dist_submit = dist_subparsers.add_parser(
        "submit",
        help="Submit a distributed processing job",
    )
    dist_submit.add_argument(
        "input",
        help="Source S3 URI (s3://bucket/prefix)",
    )
    dist_submit.add_argument(
        "-o", "--output",
        required=True,
        help="Output S3 URI (s3://bucket/prefix)",
    )
    dist_submit.add_argument(
        "-n", "--name",
        help="Job name",
    )

    # Distributed worker
    dist_worker = dist_subparsers.add_parser(
        "worker",
        help="Start a distributed worker",
    )
    dist_worker.add_argument(
        "-j", "--job",
        help="Specific job ID to process (optional)",
    )
    dist_worker.add_argument(
        "-w", "--worker-id",
        help="Worker ID (auto-generated if not specified)",
    )

    # Distributed status
    dist_status = dist_subparsers.add_parser(
        "status",
        help="Show job status",
    )
    dist_status.add_argument(
        "-j", "--job",
        help="Specific job ID (shows overall if not specified)",
    )

    # Distributed jobs
    dist_jobs = dist_subparsers.add_parser(
        "jobs",
        help="List all jobs",
    )
    dist_jobs.add_argument(
        "-s", "--status",
        help="Filter by status",
    )
    dist_jobs.add_argument(
        "-l", "--limit",
        type=int,
        default=50,
        help="Maximum jobs to list",
    )

    # Distributed workers
    dist_workers = dist_subparsers.add_parser(
        "workers",
        help="List all workers",
    )

    # Distributed cancel
    dist_cancel = dist_subparsers.add_parser(
        "cancel",
        help="Cancel a job",
    )
    dist_cancel.add_argument(
        "job",
        help="Job ID to cancel",
    )

    # Distributed cleanup
    dist_cleanup = dist_subparsers.add_parser(
        "cleanup",
        help="Clean up dead workers",
    )

    dist_parser.set_defaults(func=cmd_distributed)

    # Download command
    dl_parser = subparsers.add_parser(
        "download",
        help="Download data from public databases",
    )
    dl_subparsers = dl_parser.add_subparsers(dest="subcmd")

    # Common download arguments
    def add_download_args(parser: argparse.ArgumentParser) -> None:
        """Add common download arguments."""
        parser.add_argument(
            "-o", "--output",
            help="Output path (local path or s3://bucket/prefix)",
        )
        parser.add_argument(
            "--s3-bucket",
            help="S3 bucket for output",
        )
        parser.add_argument(
            "--s3-prefix",
            default="",
            help="S3 prefix for output",
        )
        parser.add_argument(
            "--s3-region",
            default="us-east-1",
            help="AWS region",
        )
        parser.add_argument(
            "-w", "--workers",
            type=int,
            default=10,
            help="Number of concurrent downloads",
        )
        parser.add_argument(
            "--retries",
            type=int,
            default=3,
            help="Maximum retry attempts",
        )
        parser.add_argument(
            "--no-verify",
            action="store_true",
            help="Skip checksum verification",
        )

    # PDB download
    dl_pdb = dl_subparsers.add_parser(
        "pdb",
        help="Download PDB structures",
    )
    add_download_args(dl_pdb)
    dl_pdb.add_argument(
        "--ids",
        nargs="+",
        help="Specific PDB IDs to download",
    )
    dl_pdb.add_argument(
        "--format",
        choices=["mmcif", "pdb"],
        default="mmcif",
        help="Structure format",
    )
    dl_pdb.add_argument(
        "--since",
        help="Download entries released since date (YYYY-MM-DD)",
    )
    dl_pdb.add_argument(
        "--all",
        action="store_true",
        help="Download entire PDB",
    )
    dl_pdb.add_argument(
        "--seqres",
        action="store_true",
        help="Download pdb_seqres.txt",
    )
    dl_pdb.add_argument(
        "--obsolete",
        action="store_true",
        help="Download obsolete entries list",
    )

    # UniProt download
    dl_uniprot = dl_subparsers.add_parser(
        "uniprot",
        help="Download UniProt databases",
    )
    add_download_args(dl_uniprot)
    dl_uniprot.add_argument(
        "--database",
        choices=["swissprot", "trembl", "uniref90", "uniref50", "uniref100", "bfd", "mgnify"],
        default="swissprot",
        help="Database to download",
    )
    dl_uniprot.add_argument(
        "--mgnify-version",
        default="2022_05",
        help="MGnify version for mgnify database",
    )

    # RNA download
    dl_rna = dl_subparsers.add_parser(
        "rna",
        help="Download RNA databases",
    )
    add_download_args(dl_rna)
    dl_rna.add_argument(
        "--database",
        choices=["rnacentral", "rfam", "nt"],
        default="rnacentral",
        help="Database to download",
    )
    dl_rna.add_argument(
        "--rfam-cm",
        action="store_true",
        help="Download Rfam covariance models",
    )

    # AlphaFold download
    dl_af = dl_subparsers.add_parser(
        "alphafold",
        help="Download AlphaFold predictions",
    )
    add_download_args(dl_af)
    dl_af.add_argument(
        "--ids",
        nargs="+",
        help="UniProt IDs for individual predictions",
    )
    dl_af.add_argument(
        "--proteome",
        nargs="+",
        help="Organism names or taxonomy IDs for proteome download",
    )
    dl_af.add_argument(
        "--swissprot",
        action="store_true",
        help="Download all Swiss-Prot predictions",
    )
    dl_af.add_argument(
        "--include-pae",
        action="store_true",
        help="Include PAE (Predicted Aligned Error) files",
    )
    dl_af.add_argument(
        "--version",
        default="v4",
        help="AlphaFold DB version",
    )

    # CCD download
    dl_ccd = dl_subparsers.add_parser(
        "ccd",
        help="Download Chemical Component Dictionary",
    )
    add_download_args(dl_ccd)
    dl_ccd.add_argument(
        "--components",
        nargs="+",
        help="Specific component IDs to download",
    )
    dl_ccd.add_argument(
        "--all",
        action="store_true",
        help="Download entire components.cif",
    )
    dl_ccd.add_argument(
        "--prd",
        action="store_true",
        help="Download Peptide Reference Dictionary",
    )
    dl_ccd.add_argument(
        "--aa-variants",
        action="store_true",
        help="Download amino acid variants",
    )

    dl_parser.set_defaults(func=cmd_download)

    # Parse arguments
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
