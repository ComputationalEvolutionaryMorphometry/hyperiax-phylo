import argparse

from scripts import build_data
from scripts import evaluate
from scripts import inspect_data
from scripts import run_mcmc


def test_all_script_cli_options_have_help_text():
    parsers = [
        build_data._build_parser(),
        evaluate._build_parser(),
        inspect_data._build_parser(),
        run_mcmc._build_parser(),
    ]

    for parser in parsers:
        for action in parser._actions:
            if not action.option_strings or action.dest == "help":
                continue
            assert action.help not in {None, argparse.SUPPRESS}, action.option_strings
            assert action.help.strip(), action.option_strings


def test_run_mcmc_help_hides_internal_override_dest_names():
    help_text = run_mcmc._build_parser().format_help()

    assert "OVERRIDE__" not in help_text
    assert "--num-edge-steps NUM_EDGE_STEPS" in help_text
