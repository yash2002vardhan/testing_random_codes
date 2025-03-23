class TextEditor:
    def __init__(self):
        self.history = [""]  # Initial empty state
        self.redo_stack = []

    def write(self, text):
        self.history.append(text)
        self.redo_stack.clear()  # Clear redo stack since new text is added

    def undo(self):
        if len(self.history) > 1:
            self.redo_stack.append(self.history.pop())
        else:
            print("Nothing to undo.")

    def redo(self):
        if self.redo_stack:
            self.history.append(self.redo_stack.pop())
        else:
            print("Nothing to redo.")

    def current(self):
        return self.history[-1]

text_editor = TextEditor()
text_editor.write("Hello")
text_editor.write("Hello world")
text_editor.undo()
text_editor.redo()
print(text_editor.current())
