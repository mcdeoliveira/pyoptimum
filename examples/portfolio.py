from pyoptimum import Client

print('PyOptimum Porfolio Example')

print('> Setting up problem')
# setup problem

# covariances
s1 = 0.06
s2 = 0.03
rho = .5
Q = [[s1 ** 2, s1 * s2 * rho], [s1 * s2 * rho, s2 ** 2]]

# returns
r = [.14, .08]

# initial holdings
x0 = [.1, .7]

# cashflow
cashflow = 1

# expected return
mu = .11

# bundle data
data = {
    'Q': Q,
    'r': r,
    'x0': x0,
    'mu': mu,
    'cashflow': cashflow
}

print('> Initializing client')
# initialize a client to connect to the api
username = 'demo@optimize.vicbee.net'
password = 'optimize'
client = Client(username=username, password=password)

print('> Calling API')
# make a call to the api
result = client.call('portfolio', data)

print('> Got solution')
print('  status = {}'.format(result['status']))
print('  portfolio return = {}'.format(mu))
print('  portfolio variance = {}'.format(result['obj']))
print('  positions = {}'.format(result['x']))
