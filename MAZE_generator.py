import tkinter as tk

class CoordinateRecorder:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=400, height=400, bg='white') # 20x20网格，每个方块20x20像素
        self.canvas.pack()
        self.coordinates = [] # 记录已点击的坐标
        self.size = 20 # 每个格子大小
        self.init_grid()
        self.save_button = tk.Button(self.root, text="Save Coordinates", command=self.save_coordinates)
        self.save_button.pack()

    def init_grid(self):
        for x in range(20): # 绘制20x20的网格
            for y in range(20):
                x1 = x * self.size
                y1 = y * self.size
                x2 = x1 + self.size
                y2 = y1 + self.size
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="gray")

        self.canvas.bind("<Button-1>", self.on_click) # 绑定鼠标左键点击事件

    def on_click(self, event):
        x, y = event.x // self.size, event.y // self.size # 计算被点击的网格坐标
        if (x, y) not in self.coordinates: # 如果该坐标未被点击过
            self.coordinates.append((x, y)) # 添加到集合中
            x1 = x * self.size
            y1 = y * self.size
            x2 = x1 + self.size
            y2 = y1 + self.size
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="black") # 将点击的方块变黑
            print(f"Clicked coordinates: ({x}, {y})") # 打印坐标

    def save_coordinates(self):
        print("Saved coordinates:", tuple(self.coordinates))

if __name__ == "__main__":
    root = tk.Tk()
    app = CoordinateRecorder(root)
    root.mainloop()