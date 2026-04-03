import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')


class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = list(_children)
        self.op = _op                  
        self.name = ""                 
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Tensor(self.data**other, (self,), f'**{other}')
        
        def _backward():
            self.grad += other * self.data**(other - 1) * out.grad
        
        out._backward = _backward  
        return out
    
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = np.ones_like(self.data)
        
        for node in reversed(topo):
            node._backward()
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __repr__(self):
        return f"Tensor(data={self.data:.4f}, grad={self.grad:.4f})"


class Optimizer:
    def __init__(self, params, learning_rate=0.01):
        self.params = list(params)
        self.learning_rate = learning_rate
    
    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros_like(param.data)


class SGD(Optimizer):
    def __init__(self, params, learning_rate=0.01, momentum=0.0):
        super().__init__(params, learning_rate)
        self.momentum = momentum
        self.velocities = [0.0] * len(self.params)
    
    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is not None:
                if self.momentum > 0:
                    self.velocities[i] = self.momentum * self.velocities[i] - self.learning_rate * param.grad
                    param.data += self.velocities[i]
                else:
                    param.data -= self.learning_rate * param.grad


class Adam(Optimizer):
    def __init__(self, params, learning_rate=0.01, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, learning_rate)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [0.0] * len(self.params)
        self.v = [0.0] * len(self.params)
        self.t = 0
    
    def step(self):
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is not None:
                grad = param.grad
                
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad * grad)
                
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                
                param.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2


def rosenbrock_tensor(x, y):
    term1 = (Tensor(1.0) - x) ** 2
    term2 = (y - x**2) ** 2 * 100
    return term1 + term2


def train_on_rosenbrock(optimizer_class, optimizer_params, start_point, epochs=200):
    x = Tensor(start_point[0])
    y = Tensor(start_point[1])
    
    optimizer = optimizer_class([x, y], **optimizer_params)
    
    history = {'x': [x.data.item()], 'y': [y.data.item()], 'loss': []}
    
    for epoch in range(epochs):
        loss = rosenbrock_tensor(x, y)
        loss_val = loss.data.item() if hasattr(loss.data, 'item') else loss.data
        history['loss'].append(loss_val)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        history['x'].append(x.data.item())
        history['y'].append(y.data.item())
        
        if epoch % 40 == 0:
            print(f"{optimizer_class.__name__} - Epoch {epoch}: loss = {loss_val:.6f}, pos = ({x.data.item():.4f}, {y.data.item():.4f})")
    
    return history


def plot_loss_comparison(history_sgd, history_adam):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history_sgd['loss'], label='SGD', linewidth=2, color='red', alpha=0.7)
    axes[0].plot(history_adam['loss'], label='Adam', linewidth=2, color='blue', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Comparison (Linear Scale)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].semilogy(history_sgd['loss'], label='SGD', linewidth=2, color='red', alpha=0.7)
    axes[1].semilogy(history_adam['loss'], label='Adam', linewidth=2, color='blue', alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss (log scale)')
    axes[1].set_title('Loss Comparison (Log Scale)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    min_loss_sgd = min(history_sgd['loss'])
    min_loss_adam = min(history_adam['loss'])
    final_loss_sgd = history_sgd['loss'][-1]
    final_loss_adam = history_adam['loss'][-1]
    
    info_text = f"""
    Final loss:
    • SGD:  {final_loss_sgd:.6f}
    • Adam: {final_loss_adam:.6f}
    
    Minimum loss:
    • SGD:  {min_loss_sgd:.6f}
    • Adam: {min_loss_adam:.6f}
    """
    
    axes[1].text(0.02, 0.98, info_text, transform=axes[1].transAxes, 
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('loss_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def create_3d_visualization(history_sgd, history_adam):
    x_grid = np.linspace(-2, 2, 100)
    y_grid = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = rosenbrock(X, Y)
    Z_plot = np.minimum(Z, 500)
    
    fig1 = plt.figure(figsize=(16, 8))
    
    ax1 = fig1.add_subplot(1, 2, 1, projection='3d')
    
    surf = ax1.plot_surface(X, Y, Z_plot, cmap='viridis', alpha=0.6, 
                            linewidth=0, antialiased=True)
    
    sgd_z = [rosenbrock(x, y) for x, y in zip(history_sgd['x'], history_sgd['y'])]
    sgd_z_plot = np.minimum(sgd_z, 500)
    
    ax1.plot(history_sgd['x'], history_sgd['y'], sgd_z_plot, 
             'r-', linewidth=2.5, label='SGD', alpha=0.8)
    ax1.scatter(history_sgd['x'][0], history_sgd['y'][0], sgd_z_plot[0], 
                color='red', s=80, marker='o', label='SGD Start', edgecolors='black')
    ax1.scatter(history_sgd['x'][-1], history_sgd['y'][-1], sgd_z_plot[-1], 
                color='darkred', s=150, marker='*', label='SGD End', edgecolors='black')
    
    adam_z = [rosenbrock(x, y) for x, y in zip(history_adam['x'], history_adam['y'])]
    adam_z_plot = np.minimum(adam_z, 500)
    
    ax1.plot(history_adam['x'], history_adam['y'], adam_z_plot, 
             'b-', linewidth=2.5, label='Adam', alpha=0.8)
    ax1.scatter(history_adam['x'][0], history_adam['y'][0], adam_z_plot[0], 
                color='blue', s=80, marker='o', label='Adam Start', edgecolors='black')
    ax1.scatter(history_adam['x'][-1], history_adam['y'][-1], adam_z_plot[-1], 
                color='darkblue', s=150, marker='*', label='Adam End', edgecolors='black')
    
    ax1.scatter([1.0], [1.0], [0.0], color='green', s=200, 
                marker='x', label='True Minimum (1,1)', linewidth=3)
    
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.set_zlabel('Loss', fontsize=12)
    ax1.set_title('3D: Adam vs SGD on Rosenbrock Function', fontsize=14)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.view_init(elev=30, azim=45)
    
    fig1.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='Loss Value')
    
    ax2 = fig1.add_subplot(1, 2, 2)
    
    contour = ax2.contour(X, Y, Z_plot, levels=50, cmap='viridis', alpha=0.7)
    ax2.clabel(contour, inline=True, fontsize=8)
    
    ax2.plot(history_sgd['x'], history_sgd['y'], 'r-', linewidth=2.5, label='SGD', alpha=0.8)
    ax2.plot(history_adam['x'], history_adam['y'], 'b-', linewidth=2.5, label='Adam', alpha=0.8)
    
    ax2.scatter(history_sgd['x'][0], history_sgd['y'][0], color='red', s=80, marker='o', edgecolors='black')
    ax2.scatter(history_sgd['x'][-1], history_sgd['y'][-1], color='darkred', s=150, marker='*', edgecolors='black')
    ax2.scatter(history_adam['x'][0], history_adam['y'][0], color='blue', s=80, marker='o', edgecolors='black')
    ax2.scatter(history_adam['x'][-1], history_adam['y'][-1], color='darkblue', s=150, marker='*', edgecolors='black')
    ax2.scatter([1.0], [1.0], color='green', s=200, marker='x', linewidths=3)
    
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_title('2D Contour View (Top-Down)', fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rosenbrock_3d_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig1


def create_animation(history_sgd, history_adam, save=False):
    x_grid = np.linspace(-2, 2, 100)
    y_grid = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.minimum(rosenbrock(X, Y), 500)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, linewidth=0)
    
    line_sgd, = ax.plot([], [], [], 'r-', linewidth=2, label='SGD')
    line_adam, = ax.plot([], [], [], 'b-', linewidth=2, label='Adam')
    
    point_sgd, = ax.plot([], [], [], 'ro', markersize=8)
    point_adam, = ax.plot([], [], [], 'bo', markersize=8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Loss')
    ax.set_title('Adam vs SGD: Training Animation')
    ax.legend()
    ax.view_init(elev=30, azim=45)
    
    def update(frame):
        sgd_x = history_sgd['x'][:frame+1]
        sgd_y = history_sgd['y'][:frame+1]
        sgd_z = [rosenbrock(x, y) for x, y in zip(sgd_x, sgd_y)]
        sgd_z = np.minimum(sgd_z, 500)
        line_sgd.set_data(sgd_x, sgd_y)
        line_sgd.set_3d_properties(sgd_z)
        point_sgd.set_data([sgd_x[-1]], [sgd_y[-1]])
        point_sgd.set_3d_properties([sgd_z[-1]])
        
        adam_x = history_adam['x'][:frame+1]
        adam_y = history_adam['y'][:frame+1]
        adam_z = [rosenbrock(x, y) for x, y in zip(adam_x, adam_y)]
        adam_z = np.minimum(adam_z, 500)
        line_adam.set_data(adam_x, adam_y)
        line_adam.set_3d_properties(adam_z)
        point_adam.set_data([adam_x[-1]], [adam_y[-1]])
        point_adam.set_3d_properties([adam_z[-1]])
        
        ax.set_title(f'Step {frame}: SGD loss={sgd_z[-1]:.4f}, Adam loss={adam_z[-1]:.4f}')
        return line_sgd, line_adam, point_sgd, point_adam
    
    frames = min(len(history_sgd['x']), len(history_adam['x']))
    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    
    if save:
        anim.save('adam_vs_sgd.gif', writer='pillow', fps=20)
        print("Animation saved as 'adam_vs_sgd.gif'")
    
    plt.show()
    return anim


if __name__ == "__main__":
    print("=" * 60)
    print("Adam vs SGD on Rosenbrock Function")
    print("=" * 60)
    
    start_point = (-1.5, 2.0)
    epochs = 200
    
    print("\n TRAINING SGD...")
    history_sgd = train_on_rosenbrock(
        SGD, {'learning_rate': 0.001, 'momentum': 0.0},
        start_point, epochs
    )
    
    print("\n TRAINING ADAM...")
    history_adam = train_on_rosenbrock(
        Adam, {'learning_rate': 0.01, 'betas': (0.9, 0.999)},
        start_point, epochs
    )
    
    print("\n GENERATING VISUALIZATIONS...")
    
    plot_loss_comparison(history_sgd, history_adam)
    
    create_3d_visualization(history_sgd, history_adam)
    
    print("\n GENERATING ANIMATION...")
    create_animation(history_sgd, history_adam, save=False)
    
    print("\n Done! All plots saved.")
