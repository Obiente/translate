package rooms

import "sync"

type Hub struct {
	mu sync.RWMutex
}

func NewHub() *Hub { return &Hub{} }
