import argparse


def launch_isaac(headless: bool = True):
    """Start the Isaac Sim app with cameras enabled and return it.

    Must run before anything else imports isaaclab / omni, so scripts call it at the top of `main` (never at module level of a library).
    """
    from isaaclab.app import AppLauncher

    args_cli, _ = argparse.ArgumentParser().parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    return AppLauncher(args_cli).app
