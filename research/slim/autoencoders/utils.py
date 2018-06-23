import tensorflow as tf

"""Load var from load_graph using load_session and eval() it value for assign to same variable in another graph in another session"""


def load_var(load_graph, load_session, scope, layer_scope, var_scope):
    return load_graph.get_tensor_by_name(scope + layer_scope + var_scope).eval(session=load_session)


def load_assign_wb(l_graph, l_session, graph, session, scope, layer_scope, w_scope='/weights:0', b_scope='/biases:0'):
    layer_scope = '/' + layer_scope
    w = load_var(l_graph, l_session, scope, layer_scope, w_scope)
    assign_w = tf.assign(graph.get_tensor_by_name(scope + layer_scope + w_scope), w)
    b = load_var(l_graph, l_session, scope, layer_scope, b_scope)
    assign_b = tf.assign(graph.get_tensor_by_name(scope + layer_scope + b_scope), b)
    session.run([assign_b, assign_w])


def load_assign_optimizer_vars(l_graph, l_session, graph, session, scope, layer_scope, opt_scope='/RMSProp',
                               w_scope='/weights', b_scope='/biases'):
    layer_scope = '/' + layer_scope

    opt0_w = load_var(l_graph, l_session, scope, layer_scope, w_scope + opt_scope + ':0')
    opt1_w = load_var(l_graph, l_session, scope, layer_scope, w_scope + opt_scope + '_1:0')

    assign_opt0_w = tf.assign(graph.get_tensor_by_name(scope + layer_scope + w_scope + opt_scope + ':0'), opt0_w)
    assign_opt1_w = tf.assign(graph.get_tensor_by_name(scope + layer_scope + w_scope + opt_scope + '_1:0'), opt1_w)

    opt0_b = load_var(l_graph, l_session, scope, layer_scope, b_scope + opt_scope + ':0')
    opt1_b = load_var(l_graph, l_session, scope, layer_scope, b_scope + opt_scope + '_1:0')

    assign_opt0_b = tf.assign(graph.get_tensor_by_name(scope + layer_scope + b_scope + opt_scope + ':0'), opt0_b)
    assign_opt1_b = tf.assign(graph.get_tensor_by_name(scope + layer_scope + b_scope + opt_scope + '_1:0'), opt1_b)

    session.run([assign_opt0_w, assign_opt1_w, assign_opt0_b, assign_opt1_b])


def load_layer_vars(l_graph, l_session, graph, session, scope, layer_scope, opt_scope='/RMSProp', w_scope='/weights',
                    b_scope='/biases'):
    load_assign_wb(l_graph=l_graph,
                   l_session=l_session,
                   graph=graph,
                   session=session,
                   scope=scope,
                   layer_scope=layer_scope,
                   w_scope=w_scope + ':0',
                   b_scope=b_scope + ':0')
    load_assign_optimizer_vars(l_graph=l_graph,
                               l_session=l_session,
                               graph=graph,
                               session=session,
                               scope=scope,
                               layer_scope=layer_scope,
                               opt_scope=opt_scope,
                               w_scope=w_scope,
                               b_scope=b_scope)
