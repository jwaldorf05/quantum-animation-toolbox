import sublime
import sublime_plugin


class ManimExitThenRunSceneCommand(sublime_plugin.WindowCommand):
    def run(self):
        self.window.run_command("manim_exit")
        # Slight delay ensures manim_exit finishes before run_scene starts
        sublime.set_timeout(lambda: self.window.run_command("manim_run_scene"), 100)
