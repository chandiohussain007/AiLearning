import torch
import torch.nn as nn
x= torch.randn(100,1)
y= (3* x+2)

#defining simple model

model = nn.Linear(1,1)

#Loss + Optimizer

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

#Training loop

for epoch in range(100):
  pred=model(x)
  loss = loss_fn(pred,y)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()


  if epoch % 10 == 0:
    print(f"Epoch: {epoch}, Loss: {loss.item()}")


print("Learned weight:",model.weight.item())
print("Learned bias:", model.bias.item())
