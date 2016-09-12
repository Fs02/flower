namespace flower {
    template<class T>
    void Net::add(const char *name, ILayerDef *definition){
        assert(layers_.find(name) == layers_.end());

        layers_[name] = definition->create(this, name);
    }
}
