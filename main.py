"""
Main entry point for tensor decomposition experiments.

This module provides a command-line interface to run different tensor
decomposition experiments for gait analysis and injury recovery prediction.
"""

import argparse
import logging
import sys
from pathlib import Path

from experiments.acc_vs_rd import AccVsReducedDimension
from experiments.acc_vs_rd_vs_samples import AccVsReducedDimensionVsSamples
from experiments.acc_vs_samples import AccVsSamples

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main function to run tensor decomposition experiments.
    
    Parses command-line arguments and executes the appropriate experiment.
    """
    parser = argparse.ArgumentParser(
        description="Tensor decomposition experiments for gait analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --samples      # Run accuracy vs samples experiment
  python main.py --dimension     # Run accuracy vs reduced dimension experiment
  python main.py --all           # Run combined experiment
        """,
    )
    parser.add_argument(
        "--samples",
        dest="exp",
        action="store_true",
        help="Run accuracy vs samples experiment",
    )
    parser.add_argument(
        "--dimension",
        dest="exp",
        action="store_false",
        help="Run accuracy vs reduced dimension experiment",
    )
    parser.add_argument(
        "--all",
        dest="exp_all",
        action="store_true",
        help="Run combined accuracy vs reduced dimension vs samples experiment",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="Data",
        help="Directory containing .mat data files (default: Data)",
    )
    parser.set_defaults(exp_all=False)
    parser.set_defaults(exp=False)
    args = parser.parse_args()

    # Validate data directory exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.error("Please ensure the Data directory exists with recovered10.mat, recovered50.mat, and recovered95.mat")
        sys.exit(1)

    # Check for required data files
    required_files = ["recovered10.mat", "recovered50.mat", "recovered95.mat"]
    missing_files = [f for f in required_files if not (data_dir / f).exists()]
    if missing_files:
        logger.error(f"Missing required data files: {missing_files}")
        sys.exit(1)

    try:
        if args.exp and not args.exp_all:
            logger.info("Running Accuracy vs Samples experiment...")
            experiment = AccVsSamples(data_dir=str(data_dir))
        elif not args.exp and not args.exp_all:
            logger.info("Running Accuracy vs Reduced Dimension experiment...")
            experiment = AccVsReducedDimension(data_dir=str(data_dir))
        else:
            logger.info("Running Accuracy vs Reduced Dimension vs Samples experiment...")
            experiment = AccVsReducedDimensionVsSamples(data_dir=str(data_dir))

        experiment.run_exp()
        logger.info("Experiment completed successfully!")

    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
