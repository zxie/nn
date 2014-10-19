from mom import MomentumOptimizer

# TODO Slowly increasing momentum schedule described in
# "On the importance of initialization and momentum in deep learning"
# - Sutskever et al. 2013

class NesterovOptimizer(MomentumOptimizer):

        def compute_update(self, data, labels):
            mom = self.get_mom()
            cost, _ = self.model.cost_and_grad(data, labels, back=False)
            self.update_costs(cost)

            # Update parameters with partial update for peek-ahead
            for p in self.params:
                self.params[p] = self.params[p] - mom*self.vel[p]
            _, grads = self.model.cost_and_grad(data, labels)
            # Undo updates to parameters
            for p in self.params:
                self.params[p] = self.params[p] + mom*self.vel[p]

            for p in grads:
                self.vel[p] = mom * self.vel[p] + self.alpha * grads[p]

            return cost
