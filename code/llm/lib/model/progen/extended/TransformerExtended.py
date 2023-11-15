import torch


class TransformerExtended(torch.nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        self.fitness_head = torch.nn.Linear(in_features=1024, out_features=1, bias=True)
        self.activation_function = torch.nn.LeakyReLU()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        hidden_states = self.fitness_head(hidden_states).to(torch.float32)
        output = self.activation_function(hidden_states)
        return output
